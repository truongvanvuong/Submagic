import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  RotateCw,
  Volume2,
  VolumeX,
  Maximize2,
  Layers,
  Plus,
  Trash2,
  Eye,
  EyeOff,
  Video,
  Upload,
  Subtitles,
  Sliders,
} from 'lucide-react';
import type {
  NormalizedBox,
  ProjectDetail,
  ROIConfig,
  SubtitleSegment,
  SubtitleOverlaySettings,
} from '../types/subtitle';
import { DEFAULT_SUBTITLE_SETTINGS } from '../types/subtitle';
import { getVideoStreamUrl } from '../services/api';
import { SubtitleOverlay } from './SubtitleOverlay';

export interface VideoWorkspaceProps {
  project: ProjectDetail | null;
  roiConfig: ROIConfig;
  currentTime: number;
  segments?: SubtitleSegment[];
  onTimeUpdate: (time: number) => void;
  onROIChange: (newRoiConfig: ROIConfig) => void;
  onDurationChange?: (duration: number) => void;
  onSeek?: (time: number) => void;
  onUploadClick?: () => void;
  videoRef?: React.RefObject<HTMLVideoElement>;
  readOnlyROI?: boolean;
  className?: string;
}

// Preset bounding boxes
export const PRESET_16_9: NormalizedBox = {
  x: 0.08,
  y: 0.78,
  w: 0.84,
  h: 0.18,
  label: 'Main Subtitle ROI',
};

export const PRESET_9_16: NormalizedBox = {
  x: 0.05,
  y: 0.65,
  w: 0.90,
  h: 0.20,
  label: 'Main Subtitle ROI',
};

export const DEFAULT_ROI: NormalizedBox = {
  x: 0.08,
  y: 0.78,
  w: 0.84,
  h: 0.18,
  label: 'Main Subtitle ROI',
};

type HandleType = 'nw' | 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w';

interface DragState {
  type: 'none' | 'move_main' | 'resize_main' | 'move_mask' | 'resize_mask' | 'draw_mask';
  handle?: HandleType;
  maskIndex?: number;
  startNormX: number;
  startNormY: number;
  initialBox: NormalizedBox;
}

/**
 * Format time in seconds to mm:ss or hh:mm:ss string.
 */
function formatTimecode(seconds: number): string {
  if (isNaN(seconds) || seconds < 0) return '00:00';
  const totalSec = Math.floor(seconds);
  const hrs = Math.floor(totalSec / 3600);
  const mins = Math.floor((totalSec % 3600) / 60);
  const secs = totalSec % 60;

  if (hrs > 0) {
    return `${String(hrs).padStart(2, '0')}:${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  }
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

export const VideoWorkspace: React.FC<VideoWorkspaceProps> = ({
  project,
  roiConfig,
  currentTime,
  segments = [],
  onTimeUpdate,
  onROIChange,
  onDurationChange,
  onSeek,
  onUploadClick,
  videoRef: externalVideoRef,
  readOnlyROI = false,
  className = '',
}) => {
  const internalVideoRef = useRef<HTMLVideoElement | null>(null);
  const videoElementRef = externalVideoRef || internalVideoRef;

  const containerRef = useRef<HTMLDivElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Subtitle Overlay Settings state & persistence
  const SUBTITLE_SETTINGS_STORAGE_KEY = 'submagic_subtitle_overlay_settings';
  const [subtitleSettings, setSubtitleSettings] = useState<SubtitleOverlaySettings>(() => {
    try {
      const saved = localStorage.getItem(SUBTITLE_SETTINGS_STORAGE_KEY);
      if (saved) return { ...DEFAULT_SUBTITLE_SETTINGS, ...JSON.parse(saved) };
    } catch (e) {
      // fallback
    }
    return DEFAULT_SUBTITLE_SETTINGS;
  });

  const updateSubtitleSettings = (partial: Partial<SubtitleOverlaySettings>) => {
    setSubtitleSettings((prev) => {
      const updated = { ...prev, ...partial };
      try {
        localStorage.setItem(SUBTITLE_SETTINGS_STORAGE_KEY, JSON.stringify(updated));
      } catch (e) {
        // ignore
      }
      return updated;
    });
  };

  const resetSubtitleSettings = () => {
    setSubtitleSettings(DEFAULT_SUBTITLE_SETTINGS);
    try {
      localStorage.setItem(SUBTITLE_SETTINGS_STORAGE_KEY, JSON.stringify(DEFAULT_SUBTITLE_SETTINGS));
    } catch (e) {
      // ignore
    }
  };

  const toggleSubtitlesEnabled = () => {
    updateSubtitleSettings({ enabled: !subtitleSettings.enabled });
  };

  // Playback state
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [duration, setDuration] = useState<number>(0);
  const [volume, setVolume] = useState<number>(1.0);
  const [isMuted, setIsMuted] = useState<boolean>(false);
  const [playbackRate, setPlaybackRate] = useState<number>(1.0);
  const [showROIOverlay, setShowROIOverlay] = useState<boolean>(true);
  const [isDrawingMaskMode, setIsDrawingMaskMode] = useState<boolean>(false);
  const [selectedMaskIndex, setSelectedMaskIndex] = useState<number | null>(null);
  const [videoError, setVideoError] = useState<string | null>(null);

  useEffect(() => {
    setVideoError(null);
  }, [project?.id]);

  // Letterboxed video placement within container
  const [renderRect, setRenderRect] = useState<{
    left: number;
    top: number;
    width: number;
    height: number;
  }>({ left: 0, top: 0, width: 0, height: 0 });

  // Drag interaction state
  const dragStateRef = useRef<DragState>({
    type: 'none',
    startNormX: 0,
    startNormY: 0,
    initialBox: { x: 0, y: 0, w: 0, h: 0 },
  });

  // Calculate video render rect accounting for letterbox/pillarbox
  const updateRenderRect = useCallback(() => {
    const video = videoElementRef.current;
    const container = containerRef.current;
    if (!video || !container) return;

    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;
    if (containerWidth === 0 || containerHeight === 0) return;

    const videoWidth = video.videoWidth || 16;
    const videoHeight = video.videoHeight || 9;

    const videoAspect = videoWidth / videoHeight;
    const containerAspect = containerWidth / containerHeight;

    let width = containerWidth;
    let height = containerHeight;
    let left = 0;
    let top = 0;

    if (videoAspect > containerAspect) {
      width = containerWidth;
      height = containerWidth / videoAspect;
      top = (containerHeight - height) / 2;
    } else {
      height = containerHeight;
      width = containerHeight * videoAspect;
      left = (containerWidth - width) / 2;
    }

    setRenderRect({ left, top, width, height });
  }, [videoElementRef]);

  useEffect(() => {
    window.addEventListener('resize', updateRenderRect);
    document.addEventListener('fullscreenchange', updateRenderRect);

    const container = containerRef.current;
    let observer: ResizeObserver | null = null;
    if (container && typeof ResizeObserver !== 'undefined') {
      observer = new ResizeObserver(() => {
        updateRenderRect();
      });
      observer.observe(container);
    }

    return () => {
      window.removeEventListener('resize', updateRenderRect);
      document.removeEventListener('fullscreenchange', updateRenderRect);
      if (observer) observer.disconnect();
    };
  }, [updateRenderRect]);

  // Video event handlers
  const handleLoadedMetadata = () => {
    const video = videoElementRef.current;
    if (!video) return;
    const dur = video.duration || 0;
    setDuration(dur);
    if (onDurationChange) onDurationChange(dur);
    updateRenderRect();
  };

  const handleTimeUpdate = () => {
    const video = videoElementRef.current;
    if (!video) return;
    onTimeUpdate(video.currentTime);
  };

  const togglePlayPause = () => {
    const video = videoElementRef.current;
    if (!video) return;
    if (video.paused) {
      video.play().catch(console.error);
    } else {
      video.pause();
    }
  };

  const handleSkip = (seconds: number) => {
    const video = videoElementRef.current;
    if (!video) return;
    const target = Math.max(0, Math.min(video.duration || 100, video.currentTime + seconds));
    video.currentTime = target;
    if (onSeek) onSeek(target);
  };

  const handleScrubberChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    const video = videoElementRef.current;
    if (video) {
      video.currentTime = time;
    }
    if (onSeek) onSeek(time);
  };

  const handleVolumeChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const val = parseFloat(e.target.value);
    setVolume(val);
    setIsMuted(val === 0);
    if (videoElementRef.current) {
      videoElementRef.current.volume = val;
      videoElementRef.current.muted = val === 0;
    }
  };

  const toggleMute = () => {
    const nextMuted = !isMuted;
    setIsMuted(nextMuted);
    if (videoElementRef.current) {
      videoElementRef.current.muted = nextMuted;
    }
  };

  const handleSpeedChange = (speed: number) => {
    setPlaybackRate(speed);
    if (videoElementRef.current) {
      videoElementRef.current.playbackRate = speed;
    }
  };

  const toggleFullscreen = () => {
    const container = containerRef.current;
    if (!container) return;
    if (!document.fullscreenElement) {
      container.requestFullscreen().catch(console.error);
    } else {
      document.exitFullscreen().catch(console.error);
    }
  };

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return;

      if (e.code === 'Space') {
        e.preventDefault();
        togglePlayPause();
      } else if (e.code === 'ArrowLeft') {
        e.preventDefault();
        handleSkip(e.shiftKey ? -1 : -5);
      } else if (e.code === 'ArrowRight') {
        e.preventDefault();
        handleSkip(e.shiftKey ? 1 : 5);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Quick Preset Handlers
  const applyPreset16_9 = () => {
    if (readOnlyROI) return;
    onROIChange({
      ...roiConfig,
      main_roi: { ...PRESET_16_9 },
    });
  };

  const applyPreset9_16 = () => {
    if (readOnlyROI) return;
    onROIChange({
      ...roiConfig,
      main_roi: { ...PRESET_9_16 },
    });
  };

  const resetROI = () => {
    if (readOnlyROI) return;
    onROIChange({
      main_roi: { ...DEFAULT_ROI },
      exclusion_masks: [],
    });
    setSelectedMaskIndex(null);
  };

  // Canvas drawing & interactive ROI logic
  const renderCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !showROIOverlay) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    if (w === 0 || h === 0) return;

    // Helper: draw handle
    const drawHandle = (x: number, y: number, color: string) => {
      const size = 8;
      ctx.fillStyle = color;
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.rect(x - size / 2, y - size / 2, size, size);
      ctx.fill();
      ctx.stroke();
    };

    // Draw main ROI
    const main = roiConfig.main_roi;
    const mx = main.x * w;
    const my = main.y * h;
    const mw = main.w * w;
    const mh = main.h * h;

    ctx.fillStyle = 'rgba(182, 111, 73, 0.16)';
    ctx.fillRect(mx, my, mw, mh);

    ctx.strokeStyle = '#b66f49';
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.strokeRect(mx, my, mw, mh);
    ctx.setLineDash([]);

    // Draw main label
    ctx.fillStyle = '#b66f49';
    ctx.font = 'bold 11px sans-serif';
    ctx.fillText('SUBTITLE ROI', mx + 4, my + 14);

    // Draw handles
    const handles = [
      { x: mx, y: my },
      { x: mx + mw / 2, y: my },
      { x: mx + mw, y: my },
      { x: mx + mw, y: my + mh / 2 },
      { x: mx + mw, y: my + mh },
      { x: mx + mw / 2, y: my + mh },
      { x: mx, y: my + mh },
      { x: mx, y: my + mh / 2 },
    ];
    handles.forEach((pt) => drawHandle(pt.x, pt.y, '#b66f49'));

    // Draw exclusion masks
    (roiConfig.exclusion_masks || []).forEach((mask, idx) => {
      const ex = mask.x * w;
      const ey = mask.y * h;
      const ew = mask.w * w;
      const eh = mask.h * h;

      const isSelected = selectedMaskIndex === idx;
      ctx.fillStyle = isSelected ? 'rgba(179, 65, 46, 0.35)' : 'rgba(255, 110, 0, 0.25)';
      ctx.fillRect(ex, ey, ew, eh);

      ctx.strokeStyle = isSelected ? '#b3412e' : '#ff6e00';
      ctx.lineWidth = isSelected ? 2.5 : 1.5;
      ctx.strokeRect(ex, ey, ew, eh);

      ctx.fillStyle = isSelected ? '#b3412e' : '#ff6e00';
      ctx.font = 'bold 10px sans-serif';
      ctx.fillText(mask.label || `Mask #${idx + 1}`, ex + 3, ey + 12);
    });
  }, [roiConfig, showROIOverlay, selectedMaskIndex]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (canvas && renderRect.width > 0 && renderRect.height > 0) {
      canvas.width = renderRect.width;
      canvas.height = renderRect.height;
      renderCanvas();
    }
  }, [renderRect, renderCanvas]);

  // Handle Drag / Resize interactions
  const isPointInBox = (box: NormalizedBox, nx: number, ny: number) => {
    return nx >= box.x && nx <= box.x + box.w && ny >= box.y && ny <= box.y + box.h;
  };

  const getHandleAtPixel = (
    box: NormalizedBox,
    px: number,
    py: number,
    w: number,
    h: number
  ): HandleType | null => {
    const tol = 8;
    const bx = box.x * w;
    const by = box.y * h;
    const bw = box.w * w;
    const bh = box.h * h;

    if (Math.abs(px - bx) <= tol && Math.abs(py - by) <= tol) return 'nw';
    if (Math.abs(px - (bx + bw)) <= tol && Math.abs(py - by) <= tol) return 'ne';
    if (Math.abs(px - (bx + bw)) <= tol && Math.abs(py - (by + bh)) <= tol) return 'se';
    if (Math.abs(px - bx) <= tol && Math.abs(py - (by + bh)) <= tol) return 'sw';
    if (Math.abs(px - (bx + bw / 2)) <= tol && Math.abs(py - by) <= tol) return 'n';
    if (Math.abs(px - (bx + bw / 2)) <= tol && Math.abs(py - (by + bh)) <= tol) return 's';
    if (Math.abs(px - bx) <= tol && Math.abs(py - (by + bh / 2)) <= tol) return 'w';
    if (Math.abs(px - (bx + bw)) <= tol && Math.abs(py - (by + bh / 2)) <= tol) return 'e';

    return null;
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (readOnlyROI) return;
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const pixelX = e.clientX - rect.left;
    const pixelY = e.clientY - rect.top;
    const w = canvas.width;
    const h = canvas.height;
    const normX = pixelX / w;
    const normY = pixelY / h;

    if (isDrawingMaskMode) {
      dragStateRef.current = {
        type: 'draw_mask',
        startNormX: normX,
        startNormY: normY,
        initialBox: { x: normX, y: normY, w: 0, h: 0 },
      };
      return;
    }

    const mainBox = roiConfig.main_roi;
    const handle = getHandleAtPixel(mainBox, pixelX, pixelY, w, h);
    if (handle) {
      dragStateRef.current = {
        type: 'resize_main',
        handle,
        startNormX: normX,
        startNormY: normY,
        initialBox: { ...mainBox },
      };
      return;
    }

    if (isPointInBox(mainBox, normX, normY)) {
      dragStateRef.current = {
        type: 'move_main',
        startNormX: normX,
        startNormY: normY,
        initialBox: { ...mainBox },
      };
      return;
    }

    const masks = roiConfig.exclusion_masks || [];
    for (let i = masks.length - 1; i >= 0; i--) {
      if (isPointInBox(masks[i], normX, normY)) {
        setSelectedMaskIndex(i);
        dragStateRef.current = {
          type: 'move_mask',
          maskIndex: i,
          startNormX: normX,
          startNormY: normY,
          initialBox: { ...masks[i] },
        };
        return;
      }
    }

    setSelectedMaskIndex(null);
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const pixelX = e.clientX - rect.left;
    const pixelY = e.clientY - rect.top;
    const w = canvas.width;
    const h = canvas.height;
    const normX = Math.max(0, Math.min(1, pixelX / w));
    const normY = Math.max(0, Math.min(1, pixelY / h));

    const dragState = dragStateRef.current;
    if (dragState.type !== 'none') {
      const deltaX = normX - dragState.startNormX;
      const deltaY = normY - dragState.startNormY;

      if (dragState.type === 'move_main') {
        const init = dragState.initialBox;
        const newX = Math.max(0, Math.min(1 - init.w, init.x + deltaX));
        const newY = Math.max(0, Math.min(1 - init.h, init.y + deltaY));
        onROIChange({
          ...roiConfig,
          main_roi: { ...roiConfig.main_roi, x: Number(newX.toFixed(4)), y: Number(newY.toFixed(4)) },
        });
      } else if (dragState.type === 'resize_main') {
        const init = dragState.initialBox;
        let x = init.x;
        let y = init.y;
        let bw = init.w;
        let bh = init.h;

        const hdl = dragState.handle;
        if (hdl?.includes('e')) bw = Math.max(0.05, Math.min(1 - x, init.w + deltaX));
        if (hdl?.includes('s')) bh = Math.max(0.04, Math.min(1 - y, init.h + deltaY));
        if (hdl?.includes('w')) {
          const right = init.x + init.w;
          x = Math.max(0, Math.min(right - 0.05, init.x + deltaX));
          bw = right - x;
        }
        if (hdl?.includes('n')) {
          const bottom = init.y + init.h;
          y = Math.max(0, Math.min(bottom - 0.04, init.y + deltaY));
          bh = bottom - y;
        }

        onROIChange({
          ...roiConfig,
          main_roi: {
            ...roiConfig.main_roi,
            x: Number(x.toFixed(4)),
            y: Number(y.toFixed(4)),
            w: Number(bw.toFixed(4)),
            h: Number(bh.toFixed(4)),
          },
        });
      } else if (dragState.type === 'move_mask' && dragState.maskIndex !== undefined) {
        const init = dragState.initialBox;
        const newX = Math.max(0, Math.min(1 - init.w, init.x + deltaX));
        const newY = Math.max(0, Math.min(1 - init.h, init.y + deltaY));

        const updatedMasks = [...roiConfig.exclusion_masks];
        updatedMasks[dragState.maskIndex] = {
          ...updatedMasks[dragState.maskIndex],
          x: Number(newX.toFixed(4)),
          y: Number(newY.toFixed(4)),
        };
        onROIChange({ ...roiConfig, exclusion_masks: updatedMasks });
      } else if (dragState.type === 'draw_mask') {
        const startX = dragState.startNormX;
        const startY = dragState.startNormY;
        const left = Math.min(startX, normX);
        const top = Math.min(startY, normY);
        const width = Math.abs(normX - startX);
        const height = Math.abs(normY - startY);

        dragState.initialBox = { x: left, y: top, w: width, h: height };
        renderCanvas();
      }
      return;
    }

    if (isDrawingMaskMode) {
      canvas.style.cursor = 'crosshair';
      return;
    }

    const mainBox = roiConfig.main_roi;
    const mainHandle = getHandleAtPixel(mainBox, pixelX, pixelY, w, h);
    if (mainHandle) {
      const cursors: Record<HandleType, string> = {
        nw: 'nwse-resize',
        se: 'nwse-resize',
        ne: 'nesw-resize',
        sw: 'nesw-resize',
        n: 'ns-resize',
        s: 'ns-resize',
        e: 'ew-resize',
        w: 'ew-resize',
      };
      canvas.style.cursor = cursors[mainHandle];
      return;
    }

    if (isPointInBox(mainBox, normX, normY)) {
      canvas.style.cursor = 'move';
      return;
    }

    canvas.style.cursor = 'default';
  };

  const handleMouseUp = () => {
    const dragState = dragStateRef.current;
    if (dragState.type === 'draw_mask') {
      const box = dragState.initialBox;
      if (box.w > 0.02 && box.h > 0.02) {
        const newMask: NormalizedBox = {
          x: Number(box.x.toFixed(4)),
          y: Number(box.y.toFixed(4)),
          w: Number(box.w.toFixed(4)),
          h: Number(box.h.toFixed(4)),
          label: `Mask #${(roiConfig.exclusion_masks?.length || 0) + 1}`,
        };
        const updatedMasks = [...(roiConfig.exclusion_masks || []), newMask];
        onROIChange({
          ...roiConfig,
          exclusion_masks: updatedMasks,
        });
        setSelectedMaskIndex(updatedMasks.length - 1);
      }
      setIsDrawingMaskMode(false);
    }

    dragStateRef.current = {
      type: 'none',
      startNormX: 0,
      startNormY: 0,
      initialBox: { x: 0, y: 0, w: 0, h: 0 },
    };
  };

  const videoStreamUrl = project?.id ? getVideoStreamUrl(project.id) : '';

  return (
    <div
      className={`flex flex-col bg-base-200 rounded-xl border border-blue_slate-500/20 shadow-lg overflow-hidden transition-smooth ${className}`}
    >
      {/* Top Workspace Toolbar: ROI Presets and Actions */}
      <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-2.5 bg-base-300/70 border-b border-blue_slate-500/20 text-xs">
        <div className="flex items-center gap-2 flex-wrap">
          <Layers className="w-4 h-4 text-primary" />
          <span className="font-semibold text-base-content text-xs">Khu vực ROI & Mask</span>
          <span className="text-blue_slate-500/40">|</span>

          {/* Quick Presets */}
          <div className="join border border-blue_slate-500/30 rounded-lg overflow-hidden">
            <button
              type="button"
              onClick={applyPreset16_9}
              disabled={readOnlyROI || !project}
              title="Vùng phụ đề chuẩn 16:9 (YouTube, Phim)"
              className="btn btn-xs join-item bg-base-200 hover:bg-base-300 border-none font-medium cursor-pointer"
            >
              Preset 16:9
            </button>
            <button
              type="button"
              onClick={applyPreset9_16}
              disabled={readOnlyROI || !project}
              title="Vùng phụ đề chuẩn 9:16 (TikTok, Shorts, Reels)"
              className="btn btn-xs join-item bg-base-200 hover:bg-base-300 border-none font-medium cursor-pointer"
            >
              Preset 9:16
            </button>
            <button
              type="button"
              onClick={resetROI}
              disabled={readOnlyROI || !project}
              title="Đặt lại khung ROI về mặc định"
              className="btn btn-xs join-item bg-base-200 hover:bg-base-300 border-none text-base-content/60 hover:text-base-content font-medium cursor-pointer"
            >
              Đặt lại
            </button>
          </div>
        </div>

        <div className="flex items-center gap-1.5">
          {/* Add Exclusion Mask Toggle */}
          <button
            type="button"
            onClick={() => setIsDrawingMaskMode(!isDrawingMaskMode)}
            disabled={readOnlyROI || !project}
            title="Vẽ vùng che (loại bỏ watermark/logo)"
            className={`btn btn-xs gap-1 rounded-lg font-medium cursor-pointer transition-smooth ${
              isDrawingMaskMode
                ? 'btn-warning shadow-xs'
                : 'btn-outline border-amber-500/40 text-amber-500 hover:bg-amber-500/10'
            }`}
          >
            <Plus className="w-3.5 h-3.5" />
            <span>{isDrawingMaskMode ? 'Hủy vẽ Mask' : 'Thêm Mask'}</span>
          </button>

          {/* Mask count / clear */}
          {roiConfig.exclusion_masks && roiConfig.exclusion_masks.length > 0 && (
            <button
              type="button"
              onClick={() => {
                if (readOnlyROI) return;
                onROIChange({ ...roiConfig, exclusion_masks: [] });
                setSelectedMaskIndex(null);
              }}
              title="Xóa tất cả vùng che"
              className="btn btn-xs btn-error btn-outline gap-1 rounded-lg cursor-pointer"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>Xóa ({roiConfig.exclusion_masks.length})</span>
            </button>
          )}

          {/* Overlay visibility toggle */}
          <button
            type="button"
            onClick={() => setShowROIOverlay(!showROIOverlay)}
            disabled={!project}
            title={showROIOverlay ? 'Ẩn khung ROI' : 'Hiện khung ROI'}
            className="btn btn-ghost btn-xs btn-square text-base-content/75 hover:text-primary hover:bg-base-300 rounded-lg cursor-pointer"
          >
            {showROIOverlay ? <Eye className="w-4 h-4 text-primary" /> : <EyeOff className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Main Video Viewport Container */}
      <div
        ref={containerRef}
        className="relative w-full aspect-video bg-blue_slate-100 flex items-center justify-center overflow-hidden select-none"
      >
        {videoStreamUrl ? (
          <>
            {/* HTML5 Video Element */}
            <video
              ref={videoElementRef}
              src={videoStreamUrl}
              playsInline
              preload="metadata"
              crossOrigin="anonymous"
              onTimeUpdate={handleTimeUpdate}
              onLoadedMetadata={handleLoadedMetadata}
              onError={() => {
                setVideoError(
                  'Trình duyệt không hỗ trợ giải mã trực tiếp codec của video này (ví dụ: HEVC/H.265). Bạn vẫn có thể chạy nhận diện giọng nói STT và OCR bình thường!'
                );
              }}
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
              onEnded={() => setIsPlaying(false)}
              className="w-full h-full object-contain pointer-events-auto"
            />

            {/* Interactive Canvas Overlay */}
            {showROIOverlay && renderRect.width > 0 && (
              <canvas
                ref={canvasRef}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
                style={{
                  position: 'absolute',
                  left: `${renderRect.left}px`,
                  top: `${renderRect.top}px`,
                  width: `${renderRect.width}px`,
                  height: `${renderRect.height}px`,
                }}
                className="pointer-events-auto"
              />
            )}

            {/* Synchronized Subtitle Overlay */}
            {renderRect.width > 0 && (
              <SubtitleOverlay
                videoRef={videoElementRef}
                segments={segments}
                renderRect={renderRect}
                settings={subtitleSettings}
                onPositionChange={(pos) => updateSubtitleSettings({ positionOffset: pos })}
              />
            )}
          </>
        ) : (
          /* Empty State */
          <div className="hero p-8 text-center">
            <div className="hero-content flex-col gap-3">
              <div className="w-16 h-16 rounded-2xl bg-base-300 border border-blue_slate-500/30 flex items-center justify-center text-primary shadow-lg">
                <Video className="w-8 h-8" />
              </div>
              <h3 className="text-base font-bold text-base-content tracking-tight">
                Chưa có video nào được chọn
              </h3>
              <p className="text-xs text-base-content/60 max-w-sm leading-relaxed">
                Hãy tải lên một tệp video (MP4, MKV, WebM) để bắt đầu trích xuất phụ đề tự động bằng Whisper AI hoặc Hardsub OCR.
              </p>
              {onUploadClick && (
                <button
                  type="button"
                  onClick={onUploadClick}
                  className="btn btn-sm btn-primary gap-1.5 mt-2 rounded-lg font-semibold shadow-sm cursor-pointer"
                >
                  <Upload className="w-4 h-4" />
                  <span>Tải video lên ngay</span>
                </button>
              )}
            </div>
          </div>
        )}

        {/* Drawing Mask Banner Notification */}
        {isDrawingMaskMode && (
          <div className="badge badge-warning text-warning-content absolute top-3 left-1/2 -translate-x-1/2 p-3 font-semibold shadow-md pointer-events-none animate-pulse">
            Nhấp và kéo trên khung video để vẽ vùng che (Mask)
          </div>
        )}

        {/* Unsupported Codec / Video Playback Error Notice */}
        {videoError && (
          <div className="alert alert-warning text-warning-content absolute bottom-4 left-4 right-4 z-20 shadow-lg text-xs py-2 px-3 flex items-center justify-between">
            <span>{videoError}</span>
            <button
              type="button"
              onClick={() => setVideoError(null)}
              className="btn btn-ghost btn-xs text-xs font-bold"
            >
              Đóng
            </button>
          </div>
        )}
      </div>

      {/* Bottom Video Playback Controls Bar */}
      <div className="flex flex-col px-4 py-3 bg-base-300/60 border-t border-blue_slate-500/20 gap-2.5">
        {/* Timeline Scrubber */}
        <div className="flex items-center gap-2.5">
          <span className="text-[11px] font-mono font-medium text-primary min-w-[50px] tracking-tight">
            {formatTimecode(currentTime)}
          </span>
          <div className="relative flex-1 flex items-center">
            <input
              type="range"
              min={0}
              max={duration || 100}
              step={0.05}
              value={currentTime}
              onChange={handleScrubberChange}
              disabled={!project}
              className="range range-primary range-xs w-full disabled:opacity-30 cursor-pointer"
            />
          </div>
          <span className="text-[11px] font-mono text-base-content/50 min-w-[50px] text-right tracking-tight">
            {formatTimecode(duration)}
          </span>
        </div>

        {/* Buttons Controls */}
        <div className="flex items-center justify-between text-base-content/85">
          <div className="flex items-center gap-1.5">
            {/* Play/Pause */}
            <button
              type="button"
              onClick={togglePlayPause}
              disabled={!project}
              title={isPlaying ? 'Tạm dừng (Phím Space)' : 'Phát (Phím Space)'}
              className="btn btn-primary btn-sm btn-circle shadow-md cursor-pointer transition-smooth"
            >
              {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 fill-primary-content ml-0.5" />}
            </button>

            {/* Skip -5s */}
            <button
              type="button"
              onClick={() => handleSkip(-5)}
              disabled={!project}
              title="Lùi 5 giây"
              className="btn btn-ghost btn-sm btn-square hover:bg-base-200 rounded-lg cursor-pointer text-base-content/75 hover:text-base-content"
            >
              <RotateCcw className="w-4 h-4" />
            </button>

            {/* Skip +5s */}
            <button
              type="button"
              onClick={() => handleSkip(5)}
              disabled={!project}
              title="Tua tới 5 giây"
              className="btn btn-ghost btn-sm btn-square hover:bg-base-200 rounded-lg cursor-pointer text-base-content/75 hover:text-base-content"
            >
              <RotateCw className="w-4 h-4" />
            </button>

            {/* Volume Control */}
            <div className="flex items-center gap-1.5 ml-2">
              <button
                type="button"
                onClick={toggleMute}
                disabled={!project}
                className="btn btn-ghost btn-xs btn-square hover:bg-base-200 rounded-lg cursor-pointer"
                title={isMuted ? 'Bật âm' : 'Tắt âm'}
              >
                {isMuted || volume === 0 ? <VolumeX className="w-4 h-4 text-error" /> : <Volume2 className="w-4 h-4 text-base-content/75" />}
              </button>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={isMuted ? 0 : volume}
                onChange={handleVolumeChange}
                disabled={!project}
                className="range range-xs w-16 range-primary cursor-pointer"
              />
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Subtitle CC & Settings Group */}
            <div className="flex items-center gap-1">
              {/* CC Toggle Button */}
              <button
                type="button"
                onClick={toggleSubtitlesEnabled}
                disabled={!project || segments.length === 0}
                title={subtitleSettings.enabled ? 'Tắt phụ đề (CC)' : 'Bật phụ đề (CC)'}
                className={`btn btn-xs px-2 font-bold font-mono transition-all ${
                  subtitleSettings.enabled && segments.length > 0
                    ? 'btn-primary'
                    : 'btn-ghost opacity-60'
                }`}
              >
                CC
              </button>

              {/* Subtitle Settings Dropdown (daisyUI) */}
              <div className="dropdown dropdown-top dropdown-end">
                <button
                  tabIndex={0}
                  type="button"
                  disabled={!project}
                  title="Tùy chỉnh phụ đề & vị trí"
                  className="btn btn-ghost btn-xs btn-square text-base-content/80 hover:text-primary hover:bg-base-300 rounded-lg"
                >
                  <Sliders className="w-3.5 h-3.5" />
                </button>
                <div
                  tabIndex={0}
                  className="dropdown-content z-50 card card-compact w-72 p-4 shadow-2xl bg-base-200 border border-blue_slate-500/25 mb-2 text-xs rounded-xl"
                >
                  {/* Settings Header */}
                  <div className="flex items-center justify-between border-b border-blue_slate-500/20 pb-2.5 mb-3 font-bold">
                    <span className="flex items-center gap-1.5 text-base-content text-xs tracking-tight">
                      <Subtitles className="w-4 h-4 text-primary" />
                      <span>Tùy chỉnh phụ đề</span>
                    </span>
                    <button
                      type="button"
                      onClick={resetSubtitleSettings}
                      className="text-[10px] text-base-content/50 hover:text-primary underline cursor-pointer transition-colors"
                    >
                      Mặc định
                    </button>
                  </div>

                  <div className="space-y-3">
                    {/* Vị trí phụ đề & Hướng dẫn kéo thả */}
                    <div className="p-2.5 rounded-lg bg-base-100 border border-blue_slate-500/20 flex items-center justify-between">
                      <div>
                        <div className="text-[11px] font-semibold text-base-content">Vị trí hiển thị</div>
                        <div className="text-[9px] text-base-content/60">Có thể bấm kéo trực tiếp trên video</div>
                      </div>
                      <button
                        type="button"
                        onClick={() => updateSubtitleSettings({ positionOffset: { x: 0.5, y: 0.88 }, position: 'bottom' })}
                        className="btn btn-xs btn-outline btn-primary rounded-md text-[10px] h-6 min-h-0"
                        title="Đưa phụ đề về chính giữa đáy video"
                      >
                        Đặt lại vị trí
                      </button>
                    </div>

                    {/* 1. Ngôn ngữ hiển thị */}
                    <div>
                      <label className="label-text text-[11px] font-semibold text-base-content/90 block mb-1">
                        Chế độ ngôn ngữ
                      </label>
                      <div className="join w-full grid grid-cols-3">
                        <button
                          type="button"
                          onClick={() => updateSubtitleSettings({ languageMode: 'original' })}
                          className={`btn btn-xs join-item ${
                            subtitleSettings.languageMode === 'original'
                              ? 'btn-primary font-bold'
                              : 'btn-ghost bg-base-100 border-blue_slate-500/25'
                          }`}
                        >
                          Gốc
                        </button>
                        <button
                          type="button"
                          onClick={() => updateSubtitleSettings({ languageMode: 'translated' })}
                          className={`btn btn-xs join-item ${
                            subtitleSettings.languageMode === 'translated'
                              ? 'btn-primary font-bold'
                              : 'btn-ghost bg-base-100 border-blue_slate-500/25'
                          }`}
                        >
                          Bản dịch
                        </button>
                        <button
                          type="button"
                          onClick={() => updateSubtitleSettings({ languageMode: 'bilingual' })}
                          className={`btn btn-xs join-item ${
                            subtitleSettings.languageMode === 'bilingual'
                              ? 'btn-primary font-bold'
                              : 'btn-ghost bg-base-100 border-blue_slate-500/25'
                          }`}
                        >
                          Song ngữ
                        </button>
                      </div>
                    </div>

                    {/* 2. Cỡ chữ */}
                    <div>
                      <label className="label-text text-[11px] font-semibold text-base-content/90 block mb-1">
                        Cỡ chữ
                      </label>
                      <div className="join w-full grid grid-cols-3">
                        <button
                          type="button"
                          onClick={() => updateSubtitleSettings({ fontSize: 'small' })}
                          className={`btn btn-xs join-item ${
                            subtitleSettings.fontSize === 'small'
                              ? 'btn-primary font-bold'
                              : 'btn-ghost bg-base-100 border-blue_slate-500/25'
                          }`}
                        >
                          Nhỏ
                        </button>
                        <button
                          type="button"
                          onClick={() => updateSubtitleSettings({ fontSize: 'medium' })}
                          className={`btn btn-xs join-item ${
                            subtitleSettings.fontSize === 'medium'
                              ? 'btn-primary font-bold'
                              : 'btn-ghost bg-base-100 border-blue_slate-500/25'
                          }`}
                        >
                          Vừa
                        </button>
                        <button
                          type="button"
                          onClick={() => updateSubtitleSettings({ fontSize: 'large' })}
                          className={`btn btn-xs join-item ${
                            subtitleSettings.fontSize === 'large'
                              ? 'btn-primary font-bold'
                              : 'btn-ghost bg-base-100 border-blue_slate-500/25'
                          }`}
                        >
                          Lớn
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Speed Selector */}
            <div className="join bg-base-100 border border-base-200">
              {[0.5, 1.0, 1.25, 1.5, 2.0].map((rate) => (
                <button
                  key={rate}
                  type="button"
                  onClick={() => handleSpeedChange(rate)}
                  disabled={!project}
                  className={`btn btn-xs join-item px-1.5 font-mono ${
                    playbackRate === rate
                      ? 'btn-primary font-bold'
                      : 'btn-ghost'
                  }`}
                >
                  {rate}x
                </button>
              ))}
            </div>

            {/* Fullscreen */}
            <button
              type="button"
              onClick={toggleFullscreen}
              disabled={!project}
              title="Xem toàn màn hình"
              className="btn btn-ghost btn-sm btn-square"
            >
              <Maximize2 className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VideoWorkspace;
