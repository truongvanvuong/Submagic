import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Sparkles,
  Upload,
  Plus,
  Download,
  Cpu,
  FolderOpen,
  History,
  X,
  Loader2,
  Sun,
  Moon,
} from 'lucide-react';

import type {
  ExtractionHistoryItem,
  ExtractionMode,
  HardwareInfo,
  JobStatus,
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
} from './types/subtitle';

import {
  listProjects,
  getProject,
  createProject,
  uploadVideo,
  updateROI,
  addSegment,
  updateSegment,
  deleteSegment,
  mergeSegments,
  splitSegment,
  startSTTJob,
  startOCRJob,
  startLLMJob,
  cancelJob,
  getHardwareInfo,
  getAvailableModels,
  getExtractionHistory,
  deleteExtractionHistory,
  JobWebSocketClient,
} from './services/api';

import { ControlPanel } from './components/ControlPanel';
import { ProgressBar, type LogEntry } from './components/ProgressBar';
import { ExportModal } from './components/ExportModal';
import { VideoWorkspace, PRESET_16_9 } from './components/VideoWorkspace';
import { SubtitleEditor } from './components/SubtitleEditor';
import { HistoryDrawer } from './components/HistoryDrawer';
import { ToastContainer, type ToastItem } from './components/ToastContainer';

const LAST_PROJECT_STORAGE_KEY = 'submagic_last_project_id';
const THEME_STORAGE_KEY = 'submagic_theme';

export const App: React.FC = () => {
  // Theme State ('studio-dark' | 'cool-sky')
  const [theme, setTheme] = useState<'studio-dark' | 'cool-sky'>(() => {
    const saved = localStorage.getItem(THEME_STORAGE_KEY);
    if (saved === 'cool-sky' || saved === 'warm-linen' || saved === 'warm-parchment' || saved === 'bronze') return 'cool-sky';
    return 'studio-dark';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === 'studio-dark' ? 'cool-sky' : 'studio-dark'));
  };

  // Global Project & System State
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [activeProject, setActiveProject] = useState<ProjectDetail | null>(null);
  const [segments, setSegments] = useState<SubtitleSegment[]>([]);
  const [hardwareInfo, setHardwareInfo] = useState<HardwareInfo | null>(null);
  const [modelCatalog, setModelCatalog] = useState<ModelCatalogResponse | null>(null);

  // History State
  const [historyList, setHistoryList] = useState<ExtractionHistoryItem[]>([]);
  const [isHistoryDrawerOpen, setIsHistoryDrawerOpen] = useState<boolean>(false);

  // Playback & Video Sync State
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [duration, setDuration] = useState<number>(0);

  // ROI Configuration State
  const [roiConfig, setRoiConfig] = useState<ROIConfig>({
    main_roi: { ...PRESET_16_9 },
    exclusion_masks: [],
  });

  // Processing & WebSocket Progress State
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [jobProgress, setJobProgress] = useState<number>(0);
  const [jobStatus, setJobStatus] = useState<JobStatus>('idle');
  const [jobStatusText, setJobStatusText] = useState<string>('Sẵn sàng');
  const [logs, setLogs] = useState<LogEntry[]>([]);

  // Modals
  const [isExportModalOpen, setIsExportModalOpen] = useState<boolean>(false);
  const [isNewProjectModalOpen, setIsNewProjectModalOpen] = useState<boolean>(false);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadProgress, setUploadProgress] = useState<number>(0);

  // New Project Form Inputs
  const [newProjectName, setNewProjectName] = useState<string>('');
  const [newVideoPath, setNewVideoPath] = useState<string>('');
  const [newProjectMode, setNewProjectMode] = useState<ExtractionMode>('stt');
  const [newProjectLanguage, setNewProjectLanguage] = useState<string>('auto');

  // Toasts & Browser Notification State
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  // File upload input ref
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  // WebSocket client ref
  const wsClientRef = useRef<JobWebSocketClient | null>(null);
  // Debounce ref for ROI backend updates
  const roiDebounceRef = useRef<number | null>(null);

  // Helper to dismiss toast
  const dismissToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  // Request browser Notification API permission when user interacts
  const requestNotificationPermission = useCallback(async () => {
    if (typeof window !== 'undefined' && 'Notification' in window) {
      if (Notification.permission === 'default') {
        try {
          await Notification.requestPermission();
        } catch (err) {
          console.warn('Could not request notification permission:', err);
        }
      }
    }
  }, []);

  // Show Toast (DaisyUI top-right toast + OS/Desktop Notification if backgrounded)
  const showToast = useCallback((
    type: 'success' | 'info' | 'warning' | 'error',
    title: string,
    message?: string,
    duration = 4000
  ) => {
    const id = `toast-${Date.now()}-${Math.random().toString(36).substring(2, 9)}`;
    setToasts((prev) => [...prev, { id, type, title, message, duration }]);

    // Trigger System / Browser Notification (displays on OS even if user is on another tab or app)
    if (typeof window !== 'undefined' && 'Notification' in window) {
      if (Notification.permission === 'granted') {
        try {
          const notif = new Notification(title, {
            body: message || '',
            icon: '/favicon.ico',
          });
          notif.onclick = () => {
            window.focus();
            notif.close();
          };
        } catch (e) {
          console.warn('Could not trigger Web Notification:', e);
        }
      }
    }
  }, []);

  // Helper to append a log entry
  const appendLog = useCallback((message: string, type: LogEntry['type'] = 'info') => {
    const now = new Date();
    const timestamp = now.toTimeString().split(' ')[0];
    const newEntry: LogEntry = {
      id: `${Date.now()}-${Math.random()}`,
      timestamp,
      message,
      type,
    };
    setLogs((prev) => [...prev.slice(-99), newEntry]);
  }, []);

  // Fetch extraction history list
  const refreshHistory = useCallback(async () => {
    try {
      const history = await getExtractionHistory();
      setHistoryList(history);
    } catch (err: any) {
      console.warn('Could not load history:', err);
    }
  }, []);

  // 1. Initial Load: Projects & History (Clean empty state on startup, lazy load hardware/models)
  useEffect(() => {
    const initializeApp = async () => {
      try {
        // Luôn xóa project lưu trước đó để mở app lên là trạng thái trống hoàn toàn
        try {
          localStorage.removeItem(LAST_PROJECT_STORAGE_KEY);
        } catch {
          // ignore
        }

        const [projList, history] = await Promise.all([
          listProjects().catch(() => []),
          getExtractionHistory().catch(() => []),
        ]);

        setProjects(projList);
        setHistoryList(history);

        // Mở app lên luôn trống, không tự động nạp video gần nhất
        appendLog('Chào mừng bạn đến với Sub Magic! Hãy tải lên video để bắt đầu.', 'info');

        // Nạp ngầm cấu hình phần cứng và danh mục model (non-blocking, không làm chậm mở trang)
        getHardwareInfo().then((hw) => hw && setHardwareInfo(hw)).catch(() => null);
        getAvailableModels().then((models) => models && setModelCatalog(models)).catch(() => null);
      } catch (err: any) {
        console.error('Failed to initialize app state:', err);
        appendLog(`Khởi động hệ thống: ${err?.message || 'Kiểm tra kết nối Backend'}`, 'warning');
      }
    };

    initializeApp();
  }, [appendLog]);

  // 2. Load Project Details and Segments
  const loadProjectDetails = async (projectId: string) => {
    try {
      const detail = await getProject(projectId);
      setActiveProject(detail);
      setSegments(detail.segments || []);
      if (detail.roi_config) {
        setRoiConfig(detail.roi_config);
      }
      setJobStatus(detail.status);
      setJobProgress(detail.progress);
      setIsProcessing(detail.status === 'running' || detail.status === 'queued');

      appendLog(`Đã tải dự án: "${detail.name}" (${detail.segments?.length || 0} câu phụ đề).`, 'info');
    } catch (err: any) {
      console.error(`Failed to load project ${projectId}:`, err);
      appendLog(`Lỗi tải dự án: ${err.message}`, 'error');
    }
  };

  // 3. WebSocket Integration for Active Project
  useEffect(() => {
    if (!activeProject?.id) return;

    if (wsClientRef.current) {
      wsClientRef.current.disconnect();
      wsClientRef.current = null;
    }

    const client = new JobWebSocketClient(activeProject.id);
    wsClientRef.current = client;

    // Kích hoạt kết nối WebSocket thực tế
    client.connect();

    client.on('job_status', (event: any) => {
      if (event.status) setJobStatus(event.status);
      if (event.progress !== undefined && event.progress !== null) {
        setJobProgress(event.progress);
      }
      setIsProcessing(event.status === 'running' || event.status === 'queued');
    });

    client.on('job_started', (event: WebSocketEvent) => {
      setIsProcessing(true);
      setJobStatus('running');
      setJobProgress(0);
      setJobStatusText('Bắt đầu xử lý...');
      appendLog(`Bắt đầu tác vụ ${event.job_id || ''}`, 'info');
    });

    client.on('job_progress', (event: WebSocketEvent) => {
      if (event.progress !== undefined && event.progress !== null) {
        setJobProgress(event.progress);
      }
      if (event.message) {
        setJobStatusText(event.message);
        appendLog(event.message, 'info');
      }
    });

    client.on('segment_created', (event: WebSocketEvent) => {
      if (event.progress !== undefined && event.progress !== null) {
        setJobProgress(event.progress);
      }
      if (event.message) {
        setJobStatusText(event.message);
      }
      if (event.segment) {
        const newSeg = event.segment;
        setSegments((prev) => {
          const exists = prev.some((s) => s.id === newSeg.id);
          if (exists) return prev;
          const next = [...prev, newSeg];
          next.sort((a, b) => a.start_time - b.start_time);
          return next;
        });
      }
    });

    client.on('segment_updated', (event: WebSocketEvent) => {
      if (event.progress !== undefined && event.progress !== null) {
        setJobProgress(event.progress);
      }
      if (event.message) {
        setJobStatusText(event.message);
      }
      if (event.segment) {
        const updatedSeg = event.segment;
        setSegments((prev) =>
          prev.map((s) => (s.id === updatedSeg.id ? updatedSeg : s))
        );
      }
    });

    client.on('job_completed', (event: WebSocketEvent) => {
      setIsProcessing(false);
      setJobStatus('completed');
      setJobProgress(100);
      setJobStatusText('Xử lý hoàn tất 100%!');
      appendLog(`Tác vụ hoàn thành xuất sắc! ${event.message || ''}`, 'success');
      showToast('success', 'Trích xuất phụ đề thành công!', event.message || 'Tác vụ đã hoàn tất 100%. Phụ đề đã sẵn sàng và được lưu vào lịch sử.');

      // Refresh project & history (Req 5)
      loadProjectDetails(activeProject.id);
      refreshHistory();
    });

    client.on('job_cancelled', (event: WebSocketEvent) => {
      setIsProcessing(false);
      setJobStatus('cancelled');
      setJobStatusText('Đã hủy tác vụ.');
      appendLog(`Đã dừng tác vụ: ${event.message || ''}`, 'warning');
      showToast('warning', 'Đã hủy tác vụ', event.message || 'Tiến trình trích xuất đã dừng lại.');
    });

    client.on('job_failed', (event: WebSocketEvent) => {
      setIsProcessing(false);
      setJobStatus('failed');
      const errorMsg = event.message || (event as any).error || 'Lỗi không xác định';
      setJobStatusText(`Thất bại: ${errorMsg}`);
      appendLog(`Lỗi xử lý: ${errorMsg}`, 'error');
      showToast('error', 'Trích xuất phụ đề thất bại', errorMsg);
    });

    return () => {
      client.disconnect();
    };
  }, [activeProject?.id, appendLog, refreshHistory, showToast]);

  // 3b. Heartbeat Polling Fallback during active processing
  // Ensures UI progress never freezes even if WebSocket drops or reconnects
  useEffect(() => {
    if (!isProcessing || !activeProject?.id) return;

    const interval = setInterval(async () => {
      try {
        const detail = await getProject(activeProject.id);
        if (detail.progress !== undefined && detail.progress !== null) {
          setJobProgress((prev) => Math.max(prev, detail.progress));
        }
        if (
          detail.status === 'completed' ||
          detail.status === 'failed' ||
          detail.status === 'cancelled'
        ) {
          setIsProcessing(false);
          setJobStatus(detail.status);
          if (detail.segments && detail.segments.length > 0) {
            setSegments(detail.segments);
          }
        }
      } catch (e) {
        // Silent ignore on background poll
      }
    }, 3000);

    return () => clearInterval(interval);
  }, [isProcessing, activeProject?.id]);

  // 4. ROI Change Handler
  const handleROIChange = (newConfig: ROIConfig) => {
    setRoiConfig(newConfig);

    if (activeProject) {
      if (roiDebounceRef.current) clearTimeout(roiDebounceRef.current);
      roiDebounceRef.current = window.setTimeout(async () => {
        try {
          await updateROI(activeProject.id, newConfig);
        } catch (err: any) {
          console.warn('Failed to auto-save ROI to backend:', err);
        }
      }, 1000);
    }
  };

  // 5. Video File Upload (does NOT overwrite or delete old projects - Req 5)
  const handleUploadFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    // Yêu cầu quyền thông báo hệ thống khi người dùng bắt đầu thao tác upload
    requestNotificationPermission();

    setIsUploading(true);
    setUploadProgress(0);
    const sizeMb = (file.size / (1024 * 1024)).toFixed(1);
    appendLog(`Bắt đầu tải lên video "${file.name}" (${sizeMb} MB)...`, 'info');

    try {
      const created = await uploadVideo(file, 'stt', 'auto', (percent, loaded, total) => {
        setUploadProgress(percent);
        if (percent >= 100) {
          setJobStatusText('Đã truyền tải 100%. Máy chủ đang lưu và phân tích metadata video...');
        } else {
          setJobStatusText(`Đang tải video lên: ${percent}% (${(loaded / (1024 * 1024)).toFixed(1)} / ${sizeMb} MB)`);
        }
      });
      // Prepend to projects list so existing ones stay untouched
      setProjects((prev) => [created, ...prev]);
      setActiveProject(created);
      setSegments(created.segments || []);
      if (created.roi_config) setRoiConfig(created.roi_config);

      appendLog(`Tải lên thành công 100%! Tạo dự án "${created.name}".`, 'success');
      setJobStatusText('Sẵn sàng');
      showToast('success', 'Tải video lên thành công!', `Video "${created.name}" đã sẵn sàng để trích xuất phụ đề.`);
    } catch (err: any) {
      console.error('Upload failed:', err);
      appendLog(`Tải lên thất bại: ${err.message}`, 'error');
      setJobStatusText(`Tải lên thất bại: ${err.message}`);
      showToast('error', 'Tải video thất bại', err.message || 'Không thể tải video lên máy chủ.');
    } finally {
      setIsUploading(false);
      setUploadProgress(0);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  // 6. Manual Project Creation Modal Submit
  const handleCreateProjectSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newVideoPath.trim()) return;

    try {
      const created = await createProject(
        newProjectName.trim() || undefined,
        newVideoPath.trim(),
        newProjectMode,
        newProjectLanguage
      );
      setProjects((prev) => [created, ...prev]);
      setActiveProject(created);
      setSegments(created.segments || []);
      if (created.roi_config) setRoiConfig(created.roi_config);
      setIsNewProjectModalOpen(false);
      setNewProjectName('');
      setNewVideoPath('');
      appendLog(`Đã tạo dự án "${created.name}".`, 'success');
      showToast('success', 'Tạo dự án thành công!', `Dự án "${created.name}" đã sẵn sàng.`);
    } catch (err: any) {
      appendLog(`Tạo dự án thất bại: ${err.message}`, 'error');
      showToast('error', 'Tạo dự án thất bại', err.message);
    }
  };

  // 7. Job Launch Triggers
  const handleStartSTT = async (options: STTJobRequest) => {
    if (!activeProject) return;
    requestNotificationPermission();
    try {
      setIsProcessing(true);
      setJobStatus('running');
      setJobProgress(0);
      setJobStatusText('Khởi động nhận diện giọng nói Whisper...');
      appendLog(`Khởi chạy Whisper STT (model: ${options.model_size}, lang: ${options.language || 'auto'})...`, 'info');
      await startSTTJob(options);
    } catch (err: any) {
      setIsProcessing(false);
      setJobStatus('failed');
      appendLog(`Lỗi khởi động STT: ${err.message}`, 'error');
      showToast('error', 'Không thể khởi chạy STT', err.message);
    }
  };

  const handleStartOCR = async (options: OCRJobRequest) => {
    if (!activeProject) return;
    requestNotificationPermission();
    try {
      setIsProcessing(true);
      setJobStatus('running');
      setJobProgress(0);
      setJobStatusText('Khởi động quét Hardsub video OCR...');
      options.roi_config = roiConfig;
      appendLog(`Khởi chạy OCR Hardsub (FPS: ${options.sample_fps}, lang: ${options.language || 'auto'})...`, 'info');
      await startOCRJob(options);
    } catch (err: any) {
      setIsProcessing(false);
      setJobStatus('failed');
      appendLog(`Lỗi khởi động OCR: ${err.message}`, 'error');
      showToast('error', 'Không thể khởi chạy OCR', err.message);
    }
  };

  const handleStartLLM = async (options: LLMJobRequest) => {
    if (!activeProject) return;
    requestNotificationPermission();
    try {
      setIsProcessing(true);
      setJobStatus('running');
      setJobProgress(0);
      setJobStatusText('Khởi động Gemini AI xử lý ngôn ngữ...');
      appendLog(`Khởi chạy Gemini AI (${options.fix_ocr_spelling ? 'Sửa chính tả' : `Dịch sang ${options.translate_to}`})...`, 'info');
      await startLLMJob(options);
    } catch (err: any) {
      setIsProcessing(false);
      setJobStatus('failed');
      appendLog(`Lỗi khởi động AI: ${err.message}`, 'error');
      showToast('error', 'Không thể khởi chạy AI', err.message);
    }
  };

  const handleCancelJob = async () => {
    if (!activeProject) return;
    try {
      appendLog('Đang yêu cầu dừng tác vụ...', 'warning');
      setIsProcessing(false);
      setJobStatus('cancelled');
      setJobStatusText('Đã hủy tác vụ.');
      await cancelJob(activeProject.id);
      appendLog('Đã dừng tác vụ thành công.', 'info');
    } catch (err: any) {
      setIsProcessing(false);
      setJobStatus('idle');
      appendLog(`Dừng tác vụ: ${err.message}`, 'warning');
    }
  };

  // 8. Subtitle Segment Operations
  const handleUpdateSegment = async (segmentId: string, update: SubtitleSegmentUpdate) => {
    if (!activeProject) return;
    try {
      const updated = await updateSegment(activeProject.id, segmentId, update);
      setSegments((prev) =>
        prev.map((s) => (s.id === segmentId ? { ...s, ...updated } : s))
      );
    } catch (err: any) {
      appendLog(`Lỗi cập nhật phụ đề: ${err.message}`, 'error');
    }
  };

  const handleDeleteSegment = async (segmentId: string) => {
    if (!activeProject) return;
    try {
      await deleteSegment(activeProject.id, segmentId);
      setSegments((prev) => prev.filter((s) => s.id !== segmentId));
      appendLog('Đã xóa câu phụ đề.', 'info');
    } catch (err: any) {
      appendLog(`Lỗi xóa phụ đề: ${err.message}`, 'error');
    }
  };

  const handleMergeSegments = async (firstId: string, secondId: string) => {
    if (!activeProject) return;
    try {
      const merged = await mergeSegments(activeProject.id, firstId, secondId);
      setSegments((prev) => {
        const withoutOld = prev.filter((s) => s.id !== firstId && s.id !== secondId);
        const updated = [...withoutOld, merged];
        updated.sort((a, b) => a.start_time - b.start_time);
        return updated;
      });
      appendLog('Đã gộp 2 câu phụ đề thành công.', 'success');
    } catch (err: any) {
      appendLog(`Lỗi gộp phụ đề: ${err.message}`, 'error');
    }
  };

  const handleSplitSegment = async (
    segmentId: string,
    splitTime: number,
    firstText?: string,
    secondText?: string,
    firstTranslated?: string | null,
    secondTranslated?: string | null
  ) => {
    if (!activeProject) return;
    try {
      const splitResult = await splitSegment(
        activeProject.id,
        segmentId,
        splitTime,
        firstText,
        secondText,
        firstTranslated,
        secondTranslated
      );
      setSegments((prev) => {
        const idx = prev.findIndex((s) => s.id === segmentId);
        if (idx === -1) return prev;
        const copy = [...prev];
        copy.splice(idx, 1, ...splitResult);
        copy.sort((a, b) => a.start_time - b.start_time);
        return copy;
      });
      appendLog('Đã tách câu phụ đề làm 2 thành công.', 'success');
    } catch (err: any) {
      appendLog(`Lỗi tách phụ đề: ${err.message}`, 'error');
    }
  };

  const handleAddSegment = async (atTime: number) => {
    if (!activeProject) return;
    try {
      const newSeg = await addSegment(activeProject.id, {
        start_time: Number(atTime.toFixed(3)),
        end_time: Number((atTime + 2.0).toFixed(3)),
        text: 'Câu phụ đề mới',
      });
      setSegments((prev) => {
        const updated = [...prev, newSeg];
        updated.sort((a, b) => a.start_time - b.start_time);
        return updated;
      });
      appendLog('Đã thêm một câu phụ đề mới.', 'info');
    } catch (err: any) {
      appendLog(`Lỗi thêm phụ đề: ${err.message}`, 'error');
    }
  };

  const handleAIPolishClick = () => {
    if (!activeProject) return;
    let savedKey: string | undefined = undefined;
    try {
      savedKey = localStorage.getItem('gemini_api_key') || undefined;
    } catch {
      // Ignore
    }
    if (!savedKey) {
      appendLog('Lưu ý: Chưa nhập Gemini API Key trong Control Panel. Nếu server chưa có key, tác vụ sẽ báo lỗi.', 'warning');
    }
    handleStartLLM({
      project_id: activeProject.id,
      gemini_api_key: savedKey,
      fix_ocr_spelling: true,
      batch_size: 60,
    });
  };

  // Requirement 3: Translate triggered directly with target language from daisyUI Select
  const handleTranslateClick = (targetLang: string) => {
    if (!activeProject) return;
    let savedKey: string | undefined = undefined;
    try {
      savedKey = localStorage.getItem('gemini_api_key') || undefined;
    } catch {
      // Ignore
    }
    if (!savedKey) {
      appendLog('Lưu ý: Chưa nhập Gemini API Key trong Control Panel. Vui lòng nhập API Key ở bảng điều khiển bên trái để dịch thuật.', 'warning');
    }
    handleStartLLM({
      project_id: activeProject.id,
      gemini_api_key: savedKey,
      fix_ocr_spelling: false,
      translate_to: targetLang,
      batch_size: 60,
    });
  };

  // History Drawer Action Handlers (Req 5)
  const handleSelectHistoryProject = (projectId: string) => {
    loadProjectDetails(projectId);
    setIsHistoryDrawerOpen(false);
  };

  const handleDeleteHistoryItem = async (historyId: string) => {
    try {
      await deleteExtractionHistory(historyId);
      setHistoryList((prev) => prev.filter((item) => item.id !== historyId));
      appendLog('Đã xóa bản ghi lịch sử.', 'info');
    } catch (err: any) {
      appendLog(`Lỗi xóa lịch sử: ${err.message}`, 'error');
    }
  };

  const handleDeleteHistoryBulk = async (historyIds: string[]) => {
    const results = await Promise.allSettled(historyIds.map((id) => deleteExtractionHistory(id)));
    const deletedIds = historyIds.filter((_, i) => results[i].status === 'fulfilled');
    const failed = historyIds.length - deletedIds.length;
    if (deletedIds.length > 0) {
      setHistoryList((prev) => prev.filter((item) => !deletedIds.includes(item.id)));
      appendLog(`Đã xóa ${deletedIds.length} bản ghi lịch sử.`, 'info');
      showToast('success', 'Đã xóa lịch sử', `Đã xóa ${deletedIds.length} mục đã chọn.`);
    }
    if (failed > 0) {
      appendLog(`Không thể xóa ${failed} bản ghi lịch sử.`, 'error');
      showToast('error', 'Xóa chưa hoàn tất', `${failed} mục xóa thất bại.`);
    }
  };

  return (
    <div className="flex flex-col min-h-screen bg-base-100 text-base-content font-sans antialiased">
      {/* Top Main Navigation Header */}
      <header className="sticky top-0 z-40 flex items-center justify-between px-5 py-2.5 bg-base-200/90 backdrop-blur-md border-b border-blue_slate-500/20 shadow-sm transition-smooth">
        {/* Brand / Logo */}
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-gradient-to-br from-glaucous-500 to-pale_sky-400 text-ghost_white-DEFAULT shadow-md shadow-glaucous-900/30">
            <Sparkles className="w-4.5 h-4.5 text-ghost_white-DEFAULT" />
          </div>
          <div>
            <h1 className="text-sm font-bold tracking-tight text-base-content leading-none">Sub Magic</h1>
            <p className="text-[10px] text-base-content/60 font-semibold tracking-wider uppercase mt-0.5">AI Subtitle Studio</p>
          </div>
        </div>

        {/* Center: Project Switcher & Actions */}
        <div className="flex items-center gap-2">
          {/* Project Selector Dropdown */}
          <div className="flex items-center gap-1.5 bg-base-300/60 hover:bg-base-300 border border-blue_slate-500/25 rounded-lg px-2.5 py-1 transition-smooth">
            <FolderOpen className="w-3.5 h-3.5 text-base-content/60" />
            <select
              value={activeProject?.id || ''}
              onChange={(e) => {
                if (e.target.value) {
                  loadProjectDetails(e.target.value);
                } else {
                  setActiveProject(null);
                  setSegments([]);
                  setJobStatus('idle');
                  setJobProgress(0);
                  setIsProcessing(false);
                }
              }}
              className="bg-transparent text-xs font-medium text-base-content focus:outline-none max-w-[220px] truncate cursor-pointer"
            >
              <option value="">-- Chưa chọn video --</option>
              {projects.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.name} ({p.mode.toUpperCase()})
                </option>
              ))}
            </select>
          </div>

          {/* New Project Button */}
          <button
            type="button"
            onClick={() => setIsNewProjectModalOpen(true)}
            title="Tạo dự án mới từ đường dẫn video máy chủ"
            className="btn btn-ghost btn-xs gap-1 border border-blue_slate-500/25 hover:border-primary hover:bg-base-300 cursor-pointer rounded-lg font-medium"
          >
            <Plus className="w-3.5 h-3.5 text-primary" />
            <span>Tạo mới</span>
          </button>

          {/* Upload Video Button */}
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            onChange={handleUploadFile}
            className="hidden"
          />
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
            title="Tải video trực tiếp từ máy tính"
            className="btn btn-primary btn-xs gap-1.5 min-w-[95px] shadow-sm cursor-pointer rounded-lg font-semibold"
          >
            {isUploading ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>
                  {uploadProgress > 0 && uploadProgress < 100
                    ? `${uploadProgress}%`
                    : uploadProgress >= 100
                    ? 'Đang lưu...'
                    : 'Đang tải...'}
                </span>
              </>
            ) : (
              <>
                <Upload className="w-3.5 h-3.5" />
                <span>Tải Video</span>
              </>
            )}
          </button>

          {/* History Drawer Trigger Button */}
          <button
            type="button"
            onClick={() => {
              refreshHistory();
              setIsHistoryDrawerOpen(true);
            }}
            title="Xem danh sách lịch sử các lần trích xuất"
            className="btn btn-outline btn-xs gap-1.5  hover:border-glaucous-500 hover:bg-base-300 text-base-content/85 cursor-pointer rounded-lg"
          >
            <History className="w-3.5 h-3.5 text-base-content/60" />
            <span>Lịch sử</span>
            {historyList.length > 0 && (
              <span className="badge badge-xs bg-primary text-primary-content border-none px-1.5 font-mono">
                {historyList.length}
              </span>
            )}
          </button>
        </div>

        {/* Right: Hardware Chip, Theme Toggle & Export Button */}
        <div className="flex items-center gap-2.5">
          {/* Hardware Detection Badge */}
          {hardwareInfo && (
            <div
              title={`${hardwareInfo.cpu_name} • ${
                hardwareInfo.cuda_available
                  ? hardwareInfo.gpu_names.join(', ')
                  : 'CPU inference only'
              }`}
              className="hidden md:flex items-center gap-1.5 px-2.5 py-1 text-[11px] font-mono bg-base-300/50 border border-blue_slate-500/20 rounded-lg text-base-content/75"
            >
              <Cpu className={`w-3.5 h-3.5 ${hardwareInfo.cuda_available ? 'text-accent' : 'text-base-content/40'}`} />
              <span>{hardwareInfo.cuda_available ? 'CUDA GPU' : 'CPU'}</span>
            </div>
          )}

          {/* Theme Toggle Button (Light/Dark) */}
          <button
            type="button"
            onClick={toggleTheme}
            title={theme === 'studio-dark' ? 'Chuyển sang giao diện Sáng' : 'Chuyển sang giao diện Tối'}
            className="btn btn-ghost btn-xs btn-square text-base-content/80 hover:text-primary hover:bg-base-300 rounded-lg cursor-pointer transition-smooth"
          >
            {theme === 'studio-dark' ? <Sun className="w-3.5 h-3.5 text-pale_sky-300" /> : <Moon className="w-3.5 h-3.5 text-glaucous-DEFAULT" />}
          </button>

          {/* Export Subtitles Button */}
          <button
            type="button"
            onClick={() => setIsExportModalOpen(true)}
            disabled={!activeProject || segments.length === 0}
            className="btn btn-primary btn-xs gap-1.5 shadow-sm rounded-lg font-semibold cursor-pointer disabled:cursor-not-allowed"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Xuất Phụ Đề</span>
          </button>
        </div>
      </header>

      {/* Main 2-Column Responsive Workspace Grid */}
      <main className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-4 p-4 max-w-[1700px] w-full mx-auto">
        {/* Left Column: Video Player, Control Panel, and Progress (7 Cols on desktop) */}
        <div className="lg:col-span-7 flex flex-col gap-4">
          {/* 1. Video Player & ROI Canvas */}
          <VideoWorkspace
            project={activeProject}
            roiConfig={roiConfig}
            currentTime={currentTime}
            segments={segments}
            onTimeUpdate={(t) => setCurrentTime(t)}
            onDurationChange={(d) => setDuration(d)}
            onROIChange={handleROIChange}
            onSeek={(t) => setCurrentTime(t)}
            onUploadClick={() => fileInputRef.current?.click()}
          />

          {/* 2. Extraction Control Panel (STT / OCR / Models) */}
          <ControlPanel
            project={activeProject}
            hardwareInfo={hardwareInfo}
            modelCatalog={modelCatalog}
            isProcessing={isProcessing}
            onStartSTT={handleStartSTT}
            onStartOCR={handleStartOCR}
            onStartLLM={handleStartLLM}
            onCancelJob={handleCancelJob}
            onOpenExport={() => setIsExportModalOpen(true)}
          />

          {/* 3. Real-time Progress Bar & Terminal Logs */}
          <ProgressBar
            isProcessing={isProcessing}
            progress={jobProgress}
            status={jobStatus}
            statusText={jobStatusText}
            logs={logs}
            onClearLogs={() => setLogs([])}
          />
        </div>

        {/* Right Column: Subtitle Editor (5 Cols on desktop, Full Height) */}
        <div className="lg:col-span-5 flex flex-col min-h-[580px] lg:h-[calc(100vh-80px)] lg:sticky lg:top-16">
          <SubtitleEditor
            segments={segments}
            currentTime={currentTime}
            onSeek={(t) => setCurrentTime(t)}
            onUpdateSegment={handleUpdateSegment}
            onDeleteSegment={handleDeleteSegment}
            onMergeSegments={handleMergeSegments}
            onSplitSegment={handleSplitSegment}
            onAddSegment={handleAddSegment}
            onAIPolish={handleAIPolishClick}
            onTranslate={handleTranslateClick}
            isProcessing={isProcessing}
          />
        </div>
      </main>

      {/* Export Subtitles Modal Dialog (daisyUI) */}
      <ExportModal
        isOpen={isExportModalOpen}
        onClose={() => setIsExportModalOpen(false)}
        project={activeProject}
        onExportSuccess={(filename) => {
          appendLog(`Đã tải tệp xuất: ${filename}`, 'success');
        }}
      />

      {/* Extraction History Drawer (Req 5) */}
      <HistoryDrawer
        isOpen={isHistoryDrawerOpen}
        onClose={() => setIsHistoryDrawerOpen(false)}
        historyList={historyList}
        activeProjectId={activeProject?.id}
        onSelectProject={handleSelectHistoryProject}
        onDeleteHistory={handleDeleteHistoryItem}
        onDeleteHistoryBulk={handleDeleteHistoryBulk}
      />

      {/* New Project Modal Dialog */}
      {isNewProjectModalOpen && (
        <div className="modal modal-open bg-black/60 backdrop-blur-xs">
          <div className="modal-box max-w-md p-0 overflow-hidden bg-base-200 border  rounded-xl shadow-2xl">
            <div className="flex items-center justify-between px-5 py-3.5 bg-base-300/80 border-b border-blue_slate-500/25">
              <div className="flex items-center gap-2">
                <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
                <h3 className="font-bold text-base-content text-sm tracking-tight">Tạo Dự Án Mới</h3>
              </div>
              <button
                type="button"
                onClick={() => setIsNewProjectModalOpen(false)}
                className="btn btn-ghost btn-xs btn-square rounded-lg text-base-content/70 hover:text-base-content hover:bg-base-200"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <form onSubmit={handleCreateProjectSubmit} className="p-5 flex flex-col gap-3.5 text-xs">
              <div>
                <label className="label label-text font-semibold py-1 text-base-content/85">
                  Tên dự án (Tùy chọn)
                </label>
                <input
                  type="text"
                  value={newProjectName}
                  onChange={(e) => setNewProjectName(e.target.value)}
                  placeholder="Ví dụ: Video Marketing v1"
                  className="input input-bordered input-sm w-full bg-base-100  focus:border-primary rounded-lg text-xs"
                />
              </div>

              <div>
                <label className="label label-text font-semibold py-1 text-base-content/85">
                  Đường dẫn tệp video trên máy chủ <span className="text-error font-bold">*</span>
                </label>
                <input
                  type="text"
                  required
                  value={newVideoPath}
                  onChange={(e) => setNewVideoPath(e.target.value)}
                  placeholder="Ví dụ: d:\videos\sample.mp4"
                  className="input input-bordered input-sm w-full font-mono text-xs bg-base-100  focus:border-primary rounded-lg"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="label label-text font-semibold py-1 text-base-content/85">Chế độ trích xuất</label>
                  <select
                    value={newProjectMode}
                    onChange={(e) => setNewProjectMode(e.target.value as ExtractionMode)}
                    className="select select-bordered select-sm w-full bg-base-100  focus:border-primary rounded-lg text-xs"
                  >
                    <option value="stt">STT (Âm thanh Whisper)</option>
                    <option value="ocr">OCR (Chữ viết Hardsub)</option>
                    <option value="hybrid">Kết hợp cả hai</option>
                  </select>
                </div>

                <div>
                  <label className="label label-text font-semibold py-1 text-base-content/85">Ngôn ngữ</label>
                  <select
                    value={newProjectLanguage}
                    onChange={(e) => setNewProjectLanguage(e.target.value)}
                    className="select select-bordered select-sm w-full bg-base-100  focus:border-primary rounded-lg text-xs"
                  >
                    <option value="auto">Auto (Tự động)</option>
                    <option value="vi">Tiếng Việt (vi)</option>
                    <option value="en">English (en)</option>
                    <option value="zh">中文 (zh)</option>
                    <option value="ja">日本語 (ja)</option>
                    <option value="ko">한국어 (ko)</option>
                  </select>
                </div>
              </div>

              <div className="modal-action px-0 pt-4 border-t border-blue_slate-500/20 mt-3 gap-2">
                <button
                  type="button"
                  onClick={() => setIsNewProjectModalOpen(false)}
                  className="btn btn-sm btn-ghost rounded-lg font-medium"
                >
                  Hủy
                </button>
                <button
                  type="submit"
                  className="btn btn-sm btn-primary rounded-lg font-semibold shadow-sm"
                >
                  Tạo Dự Án
                </button>
              </div>
            </form>
          </div>
          <div className="modal-backdrop bg-black/40" onClick={() => setIsNewProjectModalOpen(false)} />
        </div>
      )}

      {/* Global DaisyUI Toast Container & Multi-Notification */}
      <ToastContainer toasts={toasts} onDismiss={dismissToast} />
    </div>
  );
};

export default App;
