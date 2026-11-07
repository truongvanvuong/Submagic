import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  Search,
  Plus,
  Trash2,
  Combine,
  Scissors,
  Sparkles,
  Languages,
  Clock,
  FileText,
  Image as ImageIcon,
  X,
  ArrowUpDown,
  Maximize2,
  Check,
  AlertCircle,
} from 'lucide-react';
import type {
  SubtitleSegment,
  SubtitleSegmentUpdate,
  TranslationContext,
} from '../types/subtitle';
import { TRANSLATE_TARGET_LANGUAGES } from '../constants/languages';
import ConfirmModal from './ConfirmModal';

export interface SubtitleEditorProps {
  segments: SubtitleSegment[];
  currentTime: number;
  onSeek: (time: number) => void;
  onUpdateSegment: (segmentId: string, update: SubtitleSegmentUpdate) => Promise<void> | void;
  onDeleteSegment: (segmentId: string) => Promise<void> | void;
  onMergeSegments: (firstId: string, secondId: string) => Promise<void> | void;
  onSplitSegment: (
    segmentId: string,
    splitTime: number,
    firstText?: string,
    secondText?: string,
    firstTranslated?: string | null,
    secondTranslated?: string | null
  ) => Promise<void> | void;
  onAddSegment: (atTime: number) => Promise<void> | void;
  onAIPolish?: () => void;
  onTranslate?: (targetLang: string, context: TranslationContext) => void;
  isProcessing?: boolean;
  className?: string;
}

/**
 * Format timestamp in seconds to mm:ss.mmm string.
 */
function formatTimestamp(seconds: number): string {
  if (isNaN(seconds) || seconds < 0) return '00:00.000';
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  const ms = Math.floor((seconds - Math.floor(seconds)) * 1000);
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${String(ms).padStart(3, '0')}`;
}

/**
 * Parse mm:ss.mmm or ss.mmm string to seconds.
 */
function parseTimestamp(str: string): number | null {
  const clean = str.trim();
  const parts = clean.split(':');
  if (parts.length === 1) {
    const s = parseFloat(parts[0]);
    return isNaN(s) ? null : s;
  }
  if (parts.length === 2) {
    const m = parseFloat(parts[0]);
    const s = parseFloat(parts[1]);
    if (isNaN(m) || isNaN(s)) return null;
    return m * 60 + s;
  }
  if (parts.length === 3) {
    const h = parseFloat(parts[0]);
    const m = parseFloat(parts[1]);
    const s = parseFloat(parts[2]);
    if (isNaN(h) || isNaN(m) || isNaN(s)) return null;
    return h * 3600 + m * 60 + s;
  }
  return null;
}

// ============================================================================
// Single Subtitle Segment Item Row
// ============================================================================

interface SegmentRowProps {
  segment: SubtitleSegment;
  index: number;
  isActive: boolean;
  canMergeNext: boolean;
  onSeek: (time: number) => void;
  onUpdate: (segmentId: string, update: SubtitleSegmentUpdate) => void;
  onDelete: (segmentId: string) => void;
  onMergeWithNext: (currentId: string) => void;
  onOpenSplitDialog: (segment: SubtitleSegment) => void;
  onOpenCropModal: (segment: SubtitleSegment) => void;
}

const SegmentRow: React.FC<SegmentRowProps> = ({
  segment,
  index,
  isActive,
  canMergeNext,
  onSeek,
  onUpdate,
  onDelete,
  onMergeWithNext,
  onOpenSplitDialog,
  onOpenCropModal,
}) => {
  const [localText, setLocalText] = useState<string>(segment.text);
  const [localTranslated, setLocalTranslated] = useState<string>(segment.translated_text || '');
  const [startTimeStr, setStartTimeStr] = useState<string>(formatTimestamp(segment.start_time));
  const [endTimeStr, setEndTimeStr] = useState<string>(formatTimestamp(segment.end_time));
  const [cropFailed, setCropFailed] = useState<boolean>(false);
  const [isEditingTimes, setIsEditingTimes] = useState<boolean>(false);

  // Sync props to local state if segment updated externally
  useEffect(() => {
    setLocalText(segment.text);
    setLocalTranslated(segment.translated_text || '');
    setStartTimeStr(formatTimestamp(segment.start_time));
    setEndTimeStr(formatTimestamp(segment.end_time));
  }, [segment.text, segment.translated_text, segment.start_time, segment.end_time]);

  // Debounced auto-save for text changes
  const debounceTimerRef = useRef<number | null>(null);

  const handleTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const next = e.target.value;
    setLocalText(next);
    if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current);
    debounceTimerRef.current = window.setTimeout(() => {
      onUpdate(segment.id, { text: next });
    }, 600);
  };

  const handleTranslatedChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const next = e.target.value;
    setLocalTranslated(next);
    if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current);
    debounceTimerRef.current = window.setTimeout(() => {
      onUpdate(segment.id, { translated_text: next });
    }, 600);
  };

  const handleBlurText = () => {
    if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current);
    if (localText !== segment.text) {
      onUpdate(segment.id, { text: localText });
    }
  };

  const handleBlurTranslated = () => {
    if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current);
    if (localTranslated !== (segment.translated_text || '')) {
      onUpdate(segment.id, { translated_text: localTranslated });
    }
  };

  const handleSaveTimes = () => {
    const s = parseTimestamp(startTimeStr);
    const e = parseTimestamp(endTimeStr);
    if (s !== null && e !== null && e > s) {
      onUpdate(segment.id, { start_time: s, end_time: e });
    } else {
      // Revert if invalid
      setStartTimeStr(formatTimestamp(segment.start_time));
      setEndTimeStr(formatTimestamp(segment.end_time));
    }
    setIsEditingTimes(false);
  };

  const confidencePercent = Math.round((segment.confidence || 0) * 100);

  // Click row card to seek to subtitle start time (unless clicking interactive controls)
  const handleCardClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const target = e.target as HTMLElement;
    if (
      target.closest('button') ||
      target.closest('input') ||
      target.closest('textarea') ||
      target.closest('select') ||
      target.closest('.group\\/crop')
    ) {
      return;
    }
    onSeek(segment.start_time);
  };

  return (
    <div
      onClick={handleCardClick}
      title="Nhấp để chuyển video đến đoạn phụ đề này"
      className={`card card-bordered transition-smooth p-3.5 cursor-pointer rounded-xl ${
        isActive
          ? 'bg-glaucous-500/10 border-glaucous-500/80 shadow-md ring-1 ring-glaucous-400/50'
          : 'bg-base-100/90 border-blue_slate-500/20 hover:border-glaucous-500/40 hover:bg-base-100 hover:shadow-xs'
      }`}
    >
      {/* Top Header: Index, Source Badge, Timestamps, and Action Buttons */}
      <div className="flex items-center justify-between gap-2 mb-2">
        <div className="flex items-center gap-2 flex-wrap">
          {/* Index Number */}
          <span className="badge bg-glaucous-500 text-ghost_white-DEFAULT border-none badge-sm font-bold font-mono">
            {index + 1}
          </span>

          {/* Timestamp seeking button */}
          {!isEditingTimes ? (
            <button
              type="button"
              onClick={() => onSeek(segment.start_time)}
              title="Nhấp để chuyển video đến mốc này"
              className="btn btn-ghost btn-xs font-mono gap-1 text-glaucous-600 dark:text-pale_sky-400 hover:text-primary px-1.5 rounded"
            >
              <Clock className="w-3 h-3 text-glaucous-600 dark:text-pale_sky-400" />
              <span>{formatTimestamp(segment.start_time)}</span>
              <span className="text-base-content/40">→</span>
              <span>{formatTimestamp(segment.end_time)}</span>
            </button>
          ) : (
            <div className="flex items-center gap-1">
              <input
                type="text"
                value={startTimeStr}
                onChange={(e) => setStartTimeStr(e.target.value)}
                className="input input-xs font-mono w-20 bg-base-200 input-bordered rounded"
              />
              <span className="text-base-content/40">-</span>
              <input
                type="text"
                value={endTimeStr}
                onChange={(e) => setEndTimeStr(e.target.value)}
                className="input input-xs font-mono w-20 bg-base-200 input-bordered rounded"
              />
              <button
                type="button"
                onClick={handleSaveTimes}
                className="btn btn-success btn-xs btn-square rounded"
                title="Lưu thời gian"
              >
                <Check className="w-3 h-3 text-success-content" />
              </button>
            </div>
          )}

          {/* Edit times trigger */}
          {!isEditingTimes && (
            <button
              type="button"
              onClick={() => setIsEditingTimes(true)}
              title="Chỉnh sửa thời gian bắt đầu và kết thúc"
              className="text-[10px] text-base-content/50 hover:text-base-content underline cursor-pointer"
            >
              sửa
            </button>
          )}

          {/* Source & Confidence Badge */}
          <span
            className={`badge badge-sm font-semibold uppercase ${
              segment.source === 'ocr'
                ? 'badge-warning text-warning-content'
                : segment.source === 'stt'
                ? 'badge-info text-info-content'
                : 'badge-ghost text-base-content/70'
            }`}
          >
            {segment.source}
          </span>

          {segment.confidence < 0.85 && (
            <span
              title={`Độ tin cậy: ${confidencePercent}%`}
              className="badge badge-error badge-sm text-[10px] font-mono text-error-content"
            >
              {confidencePercent}%
            </span>
          )}
        </div>

        {/* Action Controls (Merge, Split, Delete) */}
        <div className="flex items-center gap-1">
          {canMergeNext && (
            <button
              type="button"
              onClick={() => onMergeWithNext(segment.id)}
              title="Gộp với câu tiếp theo"
              className="btn btn-ghost btn-xs btn-square text-base-content/60 hover:text-primary"
            >
              <Combine className="w-3.5 h-3.5" />
            </button>
          )}

          <button
            type="button"
            onClick={() => onOpenSplitDialog(segment)}
            title="Tách câu phụ đề làm 2"
            className="btn btn-ghost btn-xs btn-square text-base-content/60 hover:text-primary"
          >
            <Scissors className="w-3.5 h-3.5" />
          </button>

          <button
            type="button"
            onClick={() => onDelete(segment.id)}
            title="Xóa câu phụ đề"
            className="btn btn-ghost btn-xs btn-square text-base-content/40 hover:text-error"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Main Body: OCR Crop Thumbnail Preview + Text Input Area */}
      <div className="flex items-start gap-2.5">
        {/* OCR Crop Image Preview (Adjacent to subtitle text) */}
        {segment.crop_image_url && !cropFailed ? (
          <div
            onClick={() => onOpenCropModal(segment)}
            title="Xem ảnh crop phóng to để đối chiếu"
            className="relative group/crop flex-shrink-0 w-24 h-12 bg-base-200 rounded-lg border border-base-300 overflow-hidden cursor-pointer hover:border-primary transition-all flex items-center justify-center"
          >
            <img
              src={segment.crop_image_url}
              alt="OCR crop preview"
              onError={() => setCropFailed(true)}
              className="w-full h-full object-contain"
            />
            <div className="absolute inset-0 bg-ash-grey-950/40 opacity-0 group-hover/crop:opacity-100 flex items-center justify-center transition-opacity text-white">
              <Maximize2 className="w-3.5 h-3.5" />
            </div>
          </div>
        ) : segment.source === 'ocr' ? (
          <div
            title="Không có ảnh crop"
            className="flex-shrink-0 w-24 h-12 bg-base-200/50 border border-dashed border-base-300 rounded-lg flex flex-col items-center justify-center text-base-content/40 text-[10px]"
          >
            <ImageIcon className="w-4 h-4 mb-0.5 text-base-content/30" />
            <span>No Crop</span>
          </div>
        ) : null}

        {/* Subtitle Text Content (Original and Translation) */}
        <div className="flex-1 flex flex-col gap-1.5">
          {/* Original Text Input */}
          <textarea
            rows={2}
            value={localText}
            onClick={(e) => e.stopPropagation()}
            onChange={handleTextChange}
            onBlur={handleBlurText}
            placeholder="Nhập nội dung phụ đề gốc..."
            className="textarea  textarea-xs w-full text-sm leading-relaxed bg-base-200/50 border-blue_slate-500/30 text-base-content focus:border-glaucous-500 rounded-lg transition-colors font-sans"
          />

          {/* Translated Text Input */}
          <div className="flex items-center gap-1.5" onClick={(e) => e.stopPropagation()}>
            <Languages className="w-3.5 h-3.5 text-glaucous-600 dark:text-pale_sky-400 flex-shrink-0" />
            <input
              type="text"
              value={localTranslated}
              onClick={(e) => e.stopPropagation()}
              onChange={handleTranslatedChange}
              onBlur={handleBlurTranslated}
              placeholder="Bản dịch phụ đề (tùy chọn)..."
              className="input input-xs flex-1 text-xs bg-base-200/50 border-blue_slate-500/30 text-glaucous-700 dark:text-pale_sky-300 focus:border-glaucous-500 rounded-lg transition-colors font-sans"
            />
          </div>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// Main SubtitleEditor Component
// ============================================================================

export const SubtitleEditor: React.FC<SubtitleEditorProps> = ({
  segments,
  currentTime,
  onSeek,
  onUpdateSegment,
  onDeleteSegment,
  onMergeSegments,
  onSplitSegment,
  onAddSegment,
  onAIPolish,
  onTranslate,
  isProcessing = false,
  className = '',
}) => {
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [sourceFilter, setSourceFilter] = useState<'all' | 'stt' | 'ocr' | 'manual'>('all');
  const [lowConfidenceOnly, setLowConfidenceOnly] = useState<boolean>(false);
  const [autoScroll, setAutoScroll] = useState<boolean>(true);

  // Translate target language & context (persisted in localStorage)
  const [translateTarget, setTranslateTarget] = useState<string>(() => {
    return localStorage.getItem('submagic_translate_target_lang') || 'vi';
  });
  const [translationContext, setTranslationContext] = useState<TranslationContext>(() => {
    return (localStorage.getItem('submagic_translation_context') as TranslationContext) || 'casual';
  });

  // Enlarged Crop Modal state
  const [selectedCropSegment, setSelectedCropSegment] = useState<SubtitleSegment | null>(null);

  // Split Segment Dialog state
  const [splitDialogSegment, setSplitDialogSegment] = useState<SubtitleSegment | null>(null);
  const [splitTime, setSplitTime] = useState<number>(0);
  const [splitFirstText, setSplitFirstText] = useState<string>('');
  const [splitSecondText, setSplitSecondText] = useState<string>('');
  const [pendingDeleteSegmentId, setPendingDeleteSegmentId] = useState<string | null>(null);

  const listContainerRef = useRef<HTMLDivElement | null>(null);

  const handleTranslateTargetChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const lang = e.target.value;
    setTranslateTarget(lang);
    localStorage.setItem('submagic_translate_target_lang', lang);
  };

  const handleTranslationContextChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const ctx = e.target.value as TranslationContext;
    setTranslationContext(ctx);
    localStorage.setItem('submagic_translation_context', ctx);
  };

  const handleTriggerTranslate = () => {
    if (!onTranslate) return;
    onTranslate(translateTarget, translationContext);
  };

  // Identify currently active segments (supports overlapping segments)
  const activeSegmentIds = useMemo(() => {
    const ids = new Set<string>();
    for (const s of segments) {
      if (currentTime >= s.start_time && currentTime <= s.end_time) {
        ids.add(s.id);
      }
    }
    return ids;
  }, [segments, currentTime]);

  const activeSegmentIndex = useMemo(() => {
    return segments.findIndex(
      (s) => currentTime >= s.start_time && currentTime <= s.end_time
    );
  }, [segments, currentTime]);

  // Track user manual scrolling to pause auto-scroll without disturbing user reading
  const [isUserScrolling, setIsUserScrolling] = useState<boolean>(false);
  const userScrollTimeoutRef = useRef<number | null>(null);

  const handleContainerManualScroll = () => {
    setIsUserScrolling(true);
    if (userScrollTimeoutRef.current) {
      window.clearTimeout(userScrollTimeoutRef.current);
    }
    userScrollTimeoutRef.current = window.setTimeout(() => {
      setIsUserScrolling(false);
    }, 4500);
  };

  // Filtered segments
  const filteredSegments = useMemo(() => {
    return segments.filter((seg) => {
      // Search text or translation
      if (searchQuery.trim()) {
        const q = searchQuery.toLowerCase();
        const matchOrig = seg.text.toLowerCase().includes(q);
        const matchTrans = (seg.translated_text || '').toLowerCase().includes(q);
        if (!matchOrig && !matchTrans) return false;
      }

      // Source filter
      if (sourceFilter !== 'all' && seg.source !== sourceFilter) {
        return false;
      }

      // Low confidence filter (< 85%)
      if (lowConfidenceOnly && seg.confidence >= 0.85) {
        return false;
      }

      return true;
    });
  }, [segments, searchQuery, sourceFilter, lowConfidenceOnly]);

  // Auto-scroll to active segment
  useEffect(() => {
    if (!autoScroll || isUserScrolling || activeSegmentIndex === -1 || !listContainerRef.current) return;
    const container = listContainerRef.current;
    const activeElement = container.querySelector(`[data-segment-idx="${activeSegmentIndex}"]`);
    if (activeElement) {
      activeElement.scrollIntoView({
        behavior: 'smooth',
        block: 'nearest',
      });
    }
  }, [activeSegmentIndex, autoScroll, isUserScrolling]);

  // Open Split Dialog handler
  const handleOpenSplitDialog = (seg: SubtitleSegment) => {
    setSplitDialogSegment(seg);
    const defaultSplit =
      currentTime > seg.start_time + 0.1 && currentTime < seg.end_time - 0.1
        ? Number(currentTime.toFixed(3))
        : Number(((seg.start_time + seg.end_time) / 2).toFixed(3));
    setSplitTime(defaultSplit);

    const words = seg.text.split(' ');
    const mid = Math.ceil(words.length / 2);
    setSplitFirstText(words.slice(0, mid).join(' '));
    setSplitSecondText(words.slice(mid).join(' '));
  };

  const handleConfirmSplit = async () => {
    if (!splitDialogSegment) return;
    await onSplitSegment(
      splitDialogSegment.id,
      splitTime,
      splitFirstText,
      splitSecondText,
      null,
      null
    );
    setSplitDialogSegment(null);
  };

  const handleMergeWithNext = async (currentId: string) => {
    const idx = segments.findIndex((s) => s.id === currentId);
    if (idx !== -1 && idx < segments.length - 1) {
      const nextId = segments[idx + 1].id;
      await onMergeSegments(currentId, nextId);
    }
  };

  const handleSeekFromSegment = (time: number) => {
    setIsUserScrolling(false);
    onSeek(time);
  };

  return (
    <div
      className={`flex flex-col bg-base-200 rounded-xl border border-blue_slate-500/25 shadow-lg overflow-hidden h-full transition-smooth ${className}`}
    >
      {/* Top Header: Title, Segment Stats, AI Actions, Add Button */}
      <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-3 bg-base-300/80 border-b border-blue_slate-500/20">
        <div className="flex items-center gap-2">
          <FileText className="w-4 h-4 text-primary" />
          <h2 className="font-bold text-base-content text-sm tracking-tight">Subtitle Editor</h2>
          <span className="badge badge-sm bg-glaucous-500 text-ghost_white-DEFAULT border-none font-mono">
            {filteredSegments.length}/{segments.length}
          </span>
        </div>

        <div className="flex items-center gap-1.5 flex-wrap">
          {/* AI Polish Button */}
          {onAIPolish && (
            <button
              type="button"
              onClick={onAIPolish}
              disabled={isProcessing || segments.length === 0}
              title="Dùng Gemini AI để sửa lỗi chính tả phụ đề"
              className="btn btn-xs btn-outline border-blue_slate-500/30 hover:border-glaucous-500 hover:bg-base-300 text-base-content/85 gap-1.5 rounded-lg cursor-pointer"
            >
              <Sparkles className="w-3.5 h-3.5 text-glaucous-600 dark:text-pale_sky-400" />
              <span>AI Polish</span>
            </button>
          )}

          {/* AI Translate Dropdown + Context + Button */}
          {onTranslate && (
            <div className="join border border-blue_slate-500/30 rounded-lg overflow-hidden shadow-xs">
              <select
                value={translationContext}
                onChange={handleTranslationContextChange}
                disabled={isProcessing || segments.length === 0}
                className="select select-bordered select-xs join-item bg-base-100 border-none text-xs cursor-pointer max-w-[115px]"
                title="Ngữ cảnh dịch: Đời thường (Mặc định), Cổ trang, Hiện đại"
              >
                <option value="casual">🌿 Đời thường</option>
                <option value="historical">⚔️ Cổ trang</option>
                <option value="modern">🏢 Hiện đại</option>
              </select>
              <select
                value={translateTarget}
                onChange={handleTranslateTargetChange}
                disabled={isProcessing || segments.length === 0}
                className="select select-bordered select-xs join-item bg-base-100 border-none text-xs max-w-[125px] cursor-pointer"
                title="Chọn ngôn ngữ cần dịch sang"
              >
                {TRANSLATE_TARGET_LANGUAGES.map((lang) => (
                  <option key={lang.code} value={lang.code}>
                    {lang.flag ? `${lang.flag} ` : ''}{lang.label || lang.name}
                  </option>
                ))}
              </select>
              <button
                type="button"
                onClick={handleTriggerTranslate}
                disabled={isProcessing || segments.length === 0}
                title={`Bắt đầu dịch phụ đề sang ${translateTarget} (${translationContext})`}
                className="btn btn-xs bg-glaucous-600 hover:bg-glaucous-700 text-ghost_white-DEFAULT join-item gap-1 border-none cursor-pointer"
              >
                <Languages className="w-3.5 h-3.5" />
                <span>Dịch</span>
              </button>
            </div>
          )}

          {/* Add New Segment Button */}
          <button
            type="button"
            onClick={() => onAddSegment(currentTime)}
            disabled={isProcessing}
            title="Thêm một câu phụ đề mới tại mốc thời gian hiện tại"
            className="btn btn-xs btn-primary gap-1 rounded-lg font-semibold cursor-pointer shadow-sm"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>Thêm</span>
          </button>
        </div>
      </div>

      {/* Filter and Search Bar */}
      <div className="flex flex-wrap items-center gap-2 px-4 py-2.5 bg-base-300/40 border-b border-blue_slate-500/20 text-xs">
        {/* Search input */}
        <div className="relative flex-1 min-w-[150px]">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-glaucous-600 dark:text-pale_sky-400" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Tìm kiếm phụ đề..."
            className="input input-xs w-full pl-8 pr-7 bg-base-100 border-blue_slate-500/30 focus:border-glaucous-500 rounded-lg text-xs"
          />
          {searchQuery && (
            <button
              type="button"
              onClick={() => setSearchQuery('')}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-glaucous-600 dark:text-pale_sky-400 hover:text-base-content cursor-pointer"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>

        {/* Source Filter */}
        <select
          value={sourceFilter}
          onChange={(e) => setSourceFilter(e.target.value as any)}
          className="select select-xs text-xs bg-base-100 border-blue_slate-500/30 focus:border-glaucous-500 rounded-lg cursor-pointer"
        >
          <option value="all">Tất cả nguồn</option>
          <option value="stt">STT (Giọng nói)</option>
          <option value="ocr">OCR (Chữ viết)</option>
          <option value="manual">Thủ công</option>
        </select>

        {/* Low confidence filter */}
        <button
          type="button"
          onClick={() => setLowConfidenceOnly(!lowConfidenceOnly)}
          className={`btn btn-xs gap-1 rounded-lg cursor-pointer ${
            lowConfidenceOnly
              ? 'btn-error'
              : 'btn-ghost bg-base-100 border-blue_slate-500/30 text-base-content/70'
          }`}
        >
          <AlertCircle className="w-3.5 h-3.5" />
          <span>Độ tin cậy thấp</span>
        </button>

        {/* Auto-scroll toggle */}
        <button
          type="button"
          onClick={() => setAutoScroll(!autoScroll)}
          title="Tự động cuộn theo tiến trình video"
          className={`btn btn-xs gap-1 rounded-lg cursor-pointer ${
            autoScroll
              ? 'btn-primary btn-outline'
              : 'btn-ghost bg-base-100 border-blue_slate-500/30 text-base-content/50'
          }`}
        >
          <ArrowUpDown className="w-3.5 h-3.5" />
          <span>Tự cuộn</span>
        </button>
      </div>

      {/* Subtitles Scrollable List */}
      <div
        ref={listContainerRef}
        onWheel={handleContainerManualScroll}
        onTouchMove={handleContainerManualScroll}
        className="flex-1 p-3 overflow-y-auto space-y-2 bg-base-200/30 relative"
      >
        {isUserScrolling && autoScroll && (
          <div className="sticky top-1 z-30 flex justify-center pointer-events-none">
            <button
              type="button"
              onClick={() => setIsUserScrolling(false)}
              className="btn btn-xs btn-neutral shadow-md pointer-events-auto gap-1 text-[11px] animate-in fade-in"
            >
              <ArrowUpDown className="w-3 h-3 text-accent" />
              <span>Tự cuộn đang tạm dừng • Nhấp để cuộn tiếp</span>
            </button>
          </div>
        )}

        {filteredSegments.length > 0 ? (
          filteredSegments.map((seg, idx) => {
            const originalIndex = segments.findIndex((s) => s.id === seg.id);
            const isActive = activeSegmentIds.has(seg.id);
            const canMergeNext = originalIndex < segments.length - 1;

            return (
              <div key={seg.id} data-segment-idx={originalIndex}>
                <SegmentRow
                  segment={seg}
                  index={originalIndex >= 0 ? originalIndex : idx}
                  isActive={isActive}
                  canMergeNext={canMergeNext}
                  onSeek={handleSeekFromSegment}
                  onUpdate={onUpdateSegment}
                  onDelete={(id) => setPendingDeleteSegmentId(id)}
                  onMergeWithNext={handleMergeWithNext}
                  onOpenSplitDialog={handleOpenSplitDialog}
                  onOpenCropModal={(s) => setSelectedCropSegment(s)}
                />
              </div>
            );
          })
        ) : (
          <div className="flex flex-col items-center justify-center p-12 text-base-content/40">
            <FileText className="w-10 h-10 mb-2 text-base-content/30" />
            <p className="text-sm font-medium text-base-content/70">
              {segments.length === 0 ? 'Chưa có phụ đề nào' : 'Không tìm thấy phụ đề phù hợp'}
            </p>
            <p className="text-xs text-base-content/50 mt-1">
              {segments.length === 0
                ? 'Trích xuất bằng Whisper STT hoặc OCR, hoặc bấm "Thêm"'
                : 'Thử điều chỉnh từ khóa tìm kiếm hoặc bộ lọc'}
            </p>
          </div>
        )}
      </div>

      {/* Enlarged OCR Crop Modal (daisyUI modal) */}
      {selectedCropSegment && (
        <div className="modal modal-open bg-blue_slate-100/70 backdrop-blur-xs">
          <div className="modal-box max-w-lg p-0 overflow-hidden bg-base-200 border border-blue_slate-500/30 rounded-xl shadow-2xl">
            <div className="flex items-center justify-between px-5 py-3.5 bg-base-300/80 border-b border-blue_slate-500/25">
              <div className="flex items-center gap-2">
                <ImageIcon className="w-4 h-4 text-warning" />
                <h3 className="font-bold text-sm text-base-content">
                  Đối chiếu OCR Crop — Đoạn #{selectedCropSegment.index}
                </h3>
              </div>
              <button
                type="button"
                onClick={() => setSelectedCropSegment(null)}
                className="btn btn-ghost btn-xs btn-square text-base-content/70 hover:text-base-content rounded-lg"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="p-5 flex flex-col gap-4">
              <div className="w-full bg-blue_slate-100 rounded-xl p-3 flex items-center justify-center min-h-[120px] max-h-[240px] overflow-hidden border border-blue_slate-500/30">
                <img
                  src={selectedCropSegment.crop_image_url || ''}
                  alt="Enlarged crop"
                  className="max-w-full max-h-full object-contain"
                />
              </div>

              <div className="flex flex-col gap-2">
                <div className="flex items-center justify-between text-xs text-base-content/60 font-mono">
                  <span>
                    Thời gian: {formatTimestamp(selectedCropSegment.start_time)} → {formatTimestamp(selectedCropSegment.end_time)}
                  </span>
                  <span>Độ tin cậy: {Math.round(selectedCropSegment.confidence * 100)}%</span>
                </div>

                <div className="p-3 bg-base-100/80 rounded-xl border border-blue_slate-500/25 text-sm font-medium text-base-content">
                  {selectedCropSegment.text}
                </div>
              </div>
            </div>

            <div className="modal-action px-5 py-3 bg-base-300/50 m-0 border-t border-blue_slate-500/20">
              <button
                type="button"
                onClick={() => setSelectedCropSegment(null)}
                className="btn btn-sm btn-primary rounded-lg font-semibold"
              >
                Đóng
              </button>
            </div>
          </div>
          <div className="modal-backdrop bg-blue_slate-100/50" onClick={() => setSelectedCropSegment(null)} />
        </div>
      )}

      {/* Split Segment Dialog (daisyUI modal) */}
      {splitDialogSegment && (
        <div className="modal modal-open bg-blue_slate-100/70 backdrop-blur-xs">
          <div className="modal-box max-w-md p-0 overflow-hidden bg-base-200 border border-blue_slate-500/30 rounded-xl shadow-2xl">
            <div className="flex items-center justify-between px-5 py-3.5 bg-base-300/80 border-b border-blue_slate-500/25">
              <div className="flex items-center gap-2">
                <Scissors className="w-4 h-4 text-primary" />
                <h3 className="font-bold text-sm text-base-content">Tách đoạn phụ đề làm 2</h3>
              </div>
              <button
                type="button"
                onClick={() => setSplitDialogSegment(null)}
                className="btn btn-ghost btn-xs btn-square text-base-content/70 hover:text-base-content rounded-lg"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="p-5 flex flex-col gap-3.5 text-xs">
              <div>
                <label className="label label-text font-semibold py-1 text-base-content/85">
                  Mốc thời gian tách (giây)
                </label>
                <input
                  type="number"
                  step="0.05"
                  min={splitDialogSegment.start_time + 0.05}
                  max={splitDialogSegment.end_time - 0.05}
                  value={splitTime}
                  onChange={(e) => setSplitTime(parseFloat(e.target.value))}
                  className="input input-sm w-full font-mono bg-base-100 border-blue_slate-500/30 focus:border-glaucous-500 rounded-lg text-xs"
                />
                <span className="text-[11px] text-base-content/60 mt-1 block">
                  Giới hạn: {splitDialogSegment.start_time.toFixed(2)}s đến {splitDialogSegment.end_time.toFixed(2)}s
                </span>
              </div>

              <div>
                <label className="label label-text font-semibold py-1 text-base-content/85">
                  Đoạn 1 ({splitDialogSegment.start_time.toFixed(2)}s - {splitTime.toFixed(2)}s)
                </label>
                <textarea
                  rows={2}
                  value={splitFirstText}
                  onChange={(e) => setSplitFirstText(e.target.value)}
                  className="textarea textarea-bordered textarea-sm w-full bg-base-100  focus:border-glaucous-500 rounded-lg text-xs font-sans"
                />
              </div>

              <div>
                <label className="label label-text font-semibold py-1 text-base-content/85">
                  Đoạn 2 ({splitTime.toFixed(2)}s - {splitDialogSegment.end_time.toFixed(2)}s)
                </label>
                <textarea
                  rows={2}
                  value={splitSecondText}
                  onChange={(e) => setSplitSecondText(e.target.value)}
                  className="textarea textarea-bordered textarea-sm w-full bg-base-100  focus:border-glaucous-500 rounded-lg text-xs font-sans"
                />
              </div>
            </div>

            <div className="modal-action px-5 py-3.5 bg-base-300/50 m-0 border-t border-blue_slate-500/20 gap-2">
              <button
                type="button"
                onClick={() => setSplitDialogSegment(null)}
                className="btn btn-sm btn-ghost rounded-lg font-medium"
              >
                Hủy
              </button>
              <button
                type="button"
                onClick={handleConfirmSplit}
                className="btn btn-sm btn-primary rounded-lg font-semibold shadow-sm"
              >
                Xác nhận tách
              </button>
            </div>
          </div>
          <div className="modal-backdrop bg-blue_slate-100/50" onClick={() => setSplitDialogSegment(null)} />
        </div>
      )}

      {/* Delete Confirmation Modal using DaisyUI */}
      <ConfirmModal
        isOpen={!!pendingDeleteSegmentId}
        title="Xác nhận xóa câu phụ đề"
        message="Bạn có chắc chắn muốn xóa câu phụ đề này? Thao tác này sẽ xóa vĩnh viễn đoạn phụ đề đã chọn."
        confirmLabel="Xóa câu này"
        cancelLabel="Hủy"
        onConfirm={async () => {
          if (pendingDeleteSegmentId) {
            await onDeleteSegment(pendingDeleteSegmentId);
            setPendingDeleteSegmentId(null);
          }
        }}
        onCancel={() => setPendingDeleteSegmentId(null)}
      />
    </div>
  );
};

export default SubtitleEditor;
