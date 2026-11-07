/**
 * REST API client and WebSocket service for Sub Magic.
 * Provides typed communication with the FastAPI backend.
 */

import type {
  ExportRequest,
  ExtractionHistoryItem,
  ExtractionMode,
  HardwareInfo,
  JobStatusResponse,
  LLMEstimateRequest,
  LLMEstimateResponse,
  LLMJobRequest,
  ModelCatalogResponse,
  OCRJobRequest,
  ProjectDetail,
  ProjectSummary,
  ROIConfig,
  STTJobRequest,
  SubtitleSegment,
  SubtitleSegmentUpdate,
  WebSocketEvent,
} from '../types/subtitle';

// Detect development environment (Vite default port 5173 or 3000)
// To bypass WSL2 proxy bottlenecks on large video uploads and streaming,
// connect directly to backend on port 8000 when running in local dev.
const isDev =
  typeof window !== 'undefined' &&
  (window.location.port === '5173' || window.location.port === '3000');

const defaultBackendHost =
  typeof window !== 'undefined'
    ? (window.location.hostname === 'localhost' ? '127.0.0.1' : window.location.hostname)
    : '127.0.0.1';

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  (isDev ? `http://${defaultBackendHost}:8000` : '');

export const WS_BASE_URL =
  import.meta.env.VITE_WS_BASE_URL ||
  (typeof window !== 'undefined'
    ? `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${defaultBackendHost}:8000`
    : 'ws://127.0.0.1:8000');

/**
 * Custom API Error containing HTTP status code and server detail.
 */
export class ApiError extends Error {
  public status: number;
  public detail?: any;

  constructor(status: number, message: string, detail?: any) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

/**
 * Generic JSON request handler.
 */
async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  const headers = new Headers(options.headers || {});

  if (!headers.has('Content-Type') && !(options.body instanceof FormData)) {
    headers.set('Content-Type', 'application/json');
  }

  const response = await fetch(url, {
    ...options,
    headers,
  });

  if (!response.ok) {
    let errorDetail: any = null;
    let message = `API request failed with status ${response.status}`;
    try {
      const data = await response.json();
      errorDetail = data.detail || data;
      if (typeof errorDetail === 'string') {
        message = errorDetail;
      } else if (Array.isArray(errorDetail)) {
        message = errorDetail.map((e) => e.msg || JSON.stringify(e)).join(', ');
      }
    } catch {
      message = await response.text().catch(() => message);
    }
    throw new ApiError(response.status, message, errorDetail);
  }

  if (response.status === 204) {
    return {} as T;
  }

  return response.json() as Promise<T>;
}

// ============================================================================
// REST API Methods
// ============================================================================

/**
 * List all subtitle extraction projects.
 */
export async function listProjects(): Promise<ProjectSummary[]> {
  return request<ProjectSummary[]>('/api/projects');
}

/**
 * Get full project details including subtitle segments.
 */
export async function getProject(id: string): Promise<ProjectDetail> {
  return request<ProjectDetail>(`/api/projects/${id}`);
}

/**
 * Create a new project from an existing video file path on the server.
 */
export async function createProject(
  name: string | undefined,
  videoPath: string,
  mode: ExtractionMode = 'stt',
  language: string = 'vi'
): Promise<ProjectDetail> {
  return request<ProjectDetail>('/api/projects', {
    method: 'POST',
    body: JSON.stringify({
      name: name || undefined,
      video_path: videoPath,
      mode,
      language,
    }),
  });
}

/**
 * Upload a video file and initialize a new project, with real-time upload progress.
 */
export async function uploadVideo(
  file: File,
  mode: ExtractionMode = 'stt',
  language: string = 'auto',
  onProgress?: (percent: number, loaded: number, total: number) => void
): Promise<ProjectDetail> {
  return new Promise<ProjectDetail>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const url = `${API_BASE_URL}/api/projects/upload`;

    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable && onProgress) {
        const percent = Math.round((event.loaded / event.total) * 100);
        onProgress(percent, event.loaded, event.total);
      }
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const data = JSON.parse(xhr.responseText) as ProjectDetail;
          resolve(data);
        } catch {
          reject(new ApiError(xhr.status, 'Lỗi phân tích dữ liệu phản hồi từ máy chủ.'));
        }
      } else {
        let msg = `Tải lên thất bại với mã lỗi HTTP ${xhr.status}`;
        try {
          const err = JSON.parse(xhr.responseText);
          if (err.detail) {
            msg = typeof err.detail === 'string' ? err.detail : JSON.stringify(err.detail);
          }
        } catch {
          // ignore
        }
        reject(new ApiError(xhr.status, msg));
      }
    };

    xhr.onerror = () => {
      reject(new ApiError(0, 'Mất kết nối đến máy chủ khi tải lên video. Vui lòng kiểm tra lại backend.'));
    };

    xhr.ontimeout = () => {
      reject(new ApiError(0, 'Quá thời gian tải lên (timeout).'));
    };

    // 10 minutes timeout for large video files
    xhr.timeout = 600000;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('mode', mode);
    formData.append('language', language);

    xhr.open('POST', url);
    xhr.send(formData);
  });
}

/**
 * Update project Region of Interest (ROI) and Exclusion Masks.
 */
export async function updateROI(id: string, roiConfig: ROIConfig): Promise<ProjectDetail> {
  return request<ProjectDetail>(`/api/projects/${id}/roi`, {
    method: 'PUT',
    body: JSON.stringify(roiConfig),
  });
}

/**
 * Delete a project and its associated temporary files.
 */
export async function deleteProject(id: string): Promise<{ deleted: boolean; project_id: string }> {
  return request<{ deleted: boolean; project_id: string }>(`/api/projects/${id}`, {
    method: 'DELETE',
  });
}

/**
 * Bulk delete projects and their associated files safely.
 */
export async function bulkDeleteProjects(
  projectIds: string[]
): Promise<{
  deleted_count: number;
  not_found_count?: number;
  deleted: string[];
  failed?: Array<{ project_id: string; reason: string }>;
}> {
  return request<{
    deleted_count: number;
    not_found_count?: number;
    deleted: string[];
    failed?: Array<{ project_id: string; reason: string }>;
  }>(`/api/projects/bulk-delete`, {
    method: 'POST',
    body: JSON.stringify({ project_ids: projectIds }),
  });
}

/**
 * Add a manual subtitle segment to a project.
 */
export async function addSegment(
  projectId: string,
  update: SubtitleSegmentUpdate
): Promise<SubtitleSegment> {
  return request<SubtitleSegment>(`/api/projects/${projectId}/segments`, {
    method: 'POST',
    body: JSON.stringify(update),
  });
}

/**
 * Update an existing subtitle segment (text, translation, timestamps).
 */
export async function updateSegment(
  projectId: string,
  segmentId: string,
  update: SubtitleSegmentUpdate
): Promise<SubtitleSegment> {
  return request<SubtitleSegment>(`/api/projects/${projectId}/segments/${segmentId}`, {
    method: 'PATCH',
    body: JSON.stringify(update),
  });
}

/**
 * Delete a subtitle segment.
 */
export async function deleteSegment(
  projectId: string,
  segmentId: string
): Promise<{ deleted: boolean; segment_id: string }> {
  return request<{ deleted: boolean; segment_id: string }>(
    `/api/projects/${projectId}/segments/${segmentId}`,
    {
      method: 'DELETE',
    }
  );
}

/**
 * Merge two adjacent subtitle segments into a single segment.
 */
export async function mergeSegments(
  projectId: string,
  firstIdOrIds: string | [string, string] | string[],
  maybeSecondId?: string
): Promise<SubtitleSegment> {
  let firstId: string;
  let secondId: string;

  if (Array.isArray(firstIdOrIds)) {
    firstId = firstIdOrIds[0];
    secondId = firstIdOrIds[1];
  } else {
    firstId = firstIdOrIds;
    secondId = maybeSecondId || '';
  }

  const query = new URLSearchParams({
    first_id: firstId,
    second_id: secondId,
  });

  return request<SubtitleSegment>(
    `/api/projects/${projectId}/segments/merge?${query.toString()}`,
    {
      method: 'POST',
    }
  );
}

/**
 * Split a subtitle segment at a specific timestamp.
 */
export async function splitSegment(
  projectId: string,
  segmentId: string,
  splitTime: number,
  firstText?: string,
  secondText?: string,
  firstTranslated?: string | null,
  secondTranslated?: string | null
): Promise<SubtitleSegment[]> {
  return request<SubtitleSegment[]>(
    `/api/projects/${projectId}/segments/${segmentId}/split`,
    {
      method: 'POST',
      body: JSON.stringify({
        split_time: splitTime,
        first_text: firstText,
        second_text: secondText,
        first_translated: firstTranslated,
        second_translated: secondTranslated,
      }),
    }
  );
}

/**
 * Start an asynchronous Speech-to-Text extraction job.
 */
export async function startSTTJob(req: STTJobRequest): Promise<JobStatusResponse> {
  return request<JobStatusResponse>('/api/jobs/stt', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

/**
 * Start an asynchronous Hardsub OCR extraction job.
 */
export async function startOCRJob(req: OCRJobRequest): Promise<JobStatusResponse> {
  return request<JobStatusResponse>('/api/jobs/ocr', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

/**
 * Start an asynchronous LLM Post-processing (spelling / translation) job.
 */
export async function startLLMJob(req: LLMJobRequest): Promise<JobStatusResponse> {
  return request<JobStatusResponse>('/api/jobs/llm', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

/**
 * Estimate token usage and cost for an LLM translation/polish job.
 */
export async function estimateLLMJob(req: LLMEstimateRequest): Promise<LLMEstimateResponse> {
  return request<LLMEstimateResponse>('/api/jobs/estimate-llm', {
    method: 'POST',
    body: JSON.stringify(req),
  });
}

/**
 * Request cancellation of an active extraction job.
 */
export async function cancelJob(
  projectId: string
): Promise<{ project_id: string; cancelled: boolean }> {
  return request<{ project_id: string; cancelled: boolean }>(
    `/api/jobs/${projectId}/cancel`,
    {
      method: 'POST',
    }
  );
}

/**
 * Export subtitles to .SRT, .VTT, or .TXT.
 * Returns the downloaded text content, filename, and Blob for direct saving.
 */
export async function exportSubtitles(
  projectId: string,
  options: ExportRequest
): Promise<{ blob: Blob; filename: string; text: string }> {
  const url = `${API_BASE_URL}/api/projects/${projectId}/export`;

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(options),
  });

  if (!response.ok) {
    const errorText = await response.text().catch(() => 'Export request failed');
    throw new ApiError(response.status, errorText);
  }

  // Parse filename from Content-Disposition header if present
  let filename = `subtitles_${projectId}.${options.format}`;
  const disposition = response.headers.get('Content-Disposition');
  if (disposition) {
    const utf8Match = disposition.match(/filename\*=UTF-8''([^;]+)/i);
    const standardMatch = disposition.match(/filename="?([^";]+)"?/i);
    if (utf8Match && utf8Match[1]) {
      filename = decodeURIComponent(utf8Match[1]);
    } else if (standardMatch && standardMatch[1]) {
      filename = standardMatch[1];
    }
  }

  const text = await response.text();
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });

  return { blob, filename, text };
}

/**
 * Get direct streaming URL for the project video.
 * Supports HTTP 206 Partial Content range requests.
 */
export function getVideoStreamUrl(projectId: string): string {
  return `${API_BASE_URL}/api/projects/${projectId}/video`;
}

/**
 * Get video frame preview image URL at a given timestamp.
 */
export function getFramePreviewUrl(
  projectId: string,
  timestamp: number = 0.0,
  cropRoi: boolean = false
): string {
  const params = new URLSearchParams({
    timestamp: timestamp.toString(),
    crop_roi: cropRoi ? 'true' : 'false',
  });
  return `${API_BASE_URL}/api/projects/${projectId}/frame-preview?${params.toString()}`;
}

/**
 * Query detected hardware capabilities (CPU, GPU CUDA, ONNX, FFmpeg).
 */
export async function getHardwareInfo(): Promise<HardwareInfo> {
  return request<HardwareInfo>('/api/system/hardware');
}

/**
 * Query available STT and OCR model catalogs.
 */
export async function getAvailableModels(): Promise<ModelCatalogResponse> {
  return request<ModelCatalogResponse>('/api/system/models');
}

/**
 * Fetch all extraction history records.
 */
export async function getExtractionHistory(): Promise<ExtractionHistoryItem[]> {
  return request<ExtractionHistoryItem[]>('/api/history');
}

/**
 * Delete a specific extraction history item.
 */
export async function deleteExtractionHistory(historyId: string): Promise<boolean> {
  const res = await request<{ deleted: boolean; history_id: string }>(`/api/history/${historyId}`, {
    method: 'DELETE',
  });
  return res.deleted;
}

/**
 * Get direct download URL for a history item's exported subtitle file.
 */
export function getHistoryDownloadUrl(historyId: string): string {
  return `${API_BASE_URL}/api/history/${historyId}/download`;
}

// ============================================================================
// WebSocket Client
// ============================================================================

export type WebSocketEventHandler = (event: WebSocketEvent) => void;

/**
 * Robust WebSocket client managing real-time progress and subtitle streaming
 * with automated heartbeats, exponential backoff reconnection, and typed dispatching.
 */
export class JobWebSocketClient {
  private projectId: string;
  private wsUrl: string;
  private ws: WebSocket | null = null;
  private isExplicitlyClosed = false;
  private pingInterval: number | null = null;
  private reconnectTimeout: number | null = null;
  private reconnectAttempts = 0;
  private readonly maxReconnectAttempts = 8;
  private readonly initialReconnectDelay = 1000;

  private listeners: Map<string, Set<WebSocketEventHandler>> = new Map();
  private anyListeners: Set<WebSocketEventHandler> = new Set();

  constructor(projectId: string, customWsUrl?: string, autoConnect: boolean = true) {
    this.projectId = projectId;
    const base = customWsUrl || WS_BASE_URL;
    this.wsUrl = `${base.replace(/\/$/, '')}/api/jobs/ws/${projectId}`;
    if (autoConnect) {
      this.connect();
    }
  }

  /**
   * Connect to the WebSocket endpoint.
   */
  public connect(): void {
    if (this.ws && (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    this.isExplicitlyClosed = false;

    try {
      this.ws = new WebSocket(this.wsUrl);

      this.ws.onopen = () => {
        this.reconnectAttempts = 0;
        this.startHeartbeat();
      };

      this.ws.onmessage = (event: MessageEvent) => {
        this.handleMessage(event);
      };

      this.ws.onerror = (err) => {
        console.warn(`[JobWebSocketClient] Connection error for project ${this.projectId}:`, err);
      };

      this.ws.onclose = () => {
        this.stopHeartbeat();
        if (!this.isExplicitlyClosed) {
          this.scheduleReconnect();
        }
      };
    } catch (err) {
      console.error('[JobWebSocketClient] Failed to instantiate WebSocket:', err);
      this.scheduleReconnect();
    }
  }

  /**
   * Disconnect and release all resources.
   */
  public disconnect(): void {
    this.isExplicitlyClosed = true;
    this.stopHeartbeat();

    if (this.reconnectTimeout !== null) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }

    if (this.ws) {
      this.ws.onopen = null;
      this.ws.onmessage = null;
      this.ws.onerror = null;
      this.ws.onclose = null;
      if (this.ws.readyState === WebSocket.OPEN || this.ws.readyState === WebSocket.CONNECTING) {
        this.ws.close();
      }
      this.ws = null;
    }
  }

  /**
   * Register a callback for a specific event type.
   */
  public on(event: WebSocketEvent['event'] | string, handler: WebSocketEventHandler): () => void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(handler);

    return () => this.off(event, handler);
  }

  /**
   * Remove a callback for a specific event type.
   */
  public off(event: WebSocketEvent['event'] | string, handler: WebSocketEventHandler): void {
    const handlers = this.listeners.get(event);
    if (handlers) {
      handlers.delete(handler);
      if (handlers.size === 0) {
        this.listeners.delete(event);
      }
    }
  }

  /**
   * Register a callback that receives all WebSocket events.
   */
  public onAny(handler: WebSocketEventHandler): () => void {
    this.anyListeners.add(handler);
    return () => {
      this.anyListeners.delete(handler);
    };
  }

  /**
   * Check if connection is currently active.
   */
  public isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }

  /**
   * Schedule automatic reconnection using exponential backoff.
   */
  private scheduleReconnect(): void {
    if (this.isExplicitlyClosed) return;
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.warn(`[JobWebSocketClient] Max reconnection attempts reached for project ${this.projectId}`);
      return;
    }

    const delay = Math.min(
      this.initialReconnectDelay * Math.pow(1.5, this.reconnectAttempts),
      10000
    );
    this.reconnectAttempts++;

    this.reconnectTimeout = window.setTimeout(() => {
      this.connect();
    }, delay);
  }

  /**
   * Start 15-second ping heartbeat to prevent gateway timeouts.
   */
  private startHeartbeat(): void {
    this.stopHeartbeat();
    this.pingInterval = window.setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        try {
          this.ws.send('ping');
        } catch {
          // Send failed; socket will close and reconnect
        }
      }
    }, 15000);
  }

  /**
   * Stop active heartbeat.
   */
  private stopHeartbeat(): void {
    if (this.pingInterval !== null) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
  }

  /**
   * Parse incoming WebSocket frame and dispatch to registered listeners.
   */
  private handleMessage(messageEvent: MessageEvent): void {
    try {
      const data = JSON.parse(messageEvent.data);
      if (data.event === 'pong') {
        return; // Ignore heartbeat responses
      }

      const wsEvent = data as WebSocketEvent;

      // Dispatch to specific event listeners
      const handlers = this.listeners.get(wsEvent.event);
      if (handlers) {
        handlers.forEach((fn) => {
          try {
            fn(wsEvent);
          } catch (e) {
            console.error(`[JobWebSocketClient] Error in listener for event ${wsEvent.event}:`, e);
          }
        });
      }

      // Dispatch to universal listeners
      this.anyListeners.forEach((fn) => {
        try {
          fn(wsEvent);
        } catch (e) {
          console.error('[JobWebSocketClient] Error in onAny listener:', e);
        }
      });
    } catch (e) {
      console.warn('[JobWebSocketClient] Non-JSON WebSocket payload received:', messageEvent.data);
    }
  }
}
