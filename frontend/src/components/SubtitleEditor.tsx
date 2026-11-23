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
  ChevronUp,
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
  onUpdateSegment: (
    segmentId: string,
    update: SubtitleSegmentUpdate,
    notifyToast?: boolean
  ) => Promise<void> | void;
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
  onAddSegment: (atTime: number) => Promise<SubtitleSegment | void> | void;
  onAIPolish?: (targetLang?: string, context?: TranslationContext) => void;
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
  isSelected: boolean;
  isPlayingNow: boolean;
  canMergeNext: boolean;
  onSelect: (segment: SubtitleSegment) => void;
  onClose: () => void;
  onSeek: (time: number) => void;
  onUpdate: (segmentId: string, update: SubtitleSegmentUpdate, notifyToast?: boolean) => void;
  onDelete: (segmentId: string) => void;
  onMergeWithNext: (currentId: string) => void;
  onOpenSplitDialog: (segment: SubtitleSegment) => void;
  onOpenCropModal: (segment: SubtitleSegment) => void;
  onEditingFocusChange: (isFocused: boolean) => void;
}

const SegmentRow: React.FC<SegmentRowProps> = ({
  segment,
  index,
  isSelected,
  isPlayingNow,
  canMergeNext,
  onSelect,
  onClose,
  onSeek,
  onUpdate,
  onDelete,
  onMergeWithNext,
  onOpenSplitDialog,
  onOpenCropModal,
  onEditingFocusChange,
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

  // Reset time editing mode if row is deselected
  useEffect(() => {
    if (!isSelected && isEditingTimes) {
      setIsEditingTimes(false);
    }
  }, [isSelected, isEditingTimes]);

  // Debounced auto-save for text changes
  const debounceTimerRef = useRef<number | null>(null);

  const handleTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const next = e.target.value;
    setLocalText(next);
    if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current);
    debounceTimerRef.current = window.setTimeout(() => {
      onUpdate(segment.id, { text: next }, false);
    }, 600);
  };

  const handleTranslatedChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const next = e.target.value;
    setLocalTranslated(next);
    if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current);
    debounceTimerRef.current = window.setTimeout(() => {
      onUpdate(segment.id, { translated_text: next }, false);
    }, 600);
  };

  const handleBlurText = () => {
    onEditingFocusChange(false);
    if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current);
    if (localText !== segment.text) {
      onUpdate(segment.id, { text: localText }, true);
    }
  };

  const handleBlurTranslated = () => {
    onEditingFocusChange(false);
    if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current);
    if (localTranslated !== (segment.translated_text || '')) {
      onUpdate(segment.id, { translated_text: localTranslated }, true);
    }
  };

  const handleSaveTimes = () => {
    const s = parseTimestamp(startTimeStr);
    const e = parseTimestamp(endTimeStr);
    if (s !== null && e !== null && e > s) {
      onUpdate(segment.id, { start_time: s, end_time: e }, true);
    } else {
      // Revert if invalid
      setStartTimeStr(formatTimestamp(segment.start_time));
      setEndTimeStr(formatTimestamp(segment.end_time));
    }
    setIsEditingTimes(false);
    onEditingFocusChange(false);
  };

  const handleCancelTimes = () => {
    setStartTimeStr(formatTimestamp(segment.start_time));
    setEndTimeStr(formatTimestamp(segment.end_time));
    setIsEditingTimes(false);
    onEditingFocusChange(false);
  };

  const confidencePercent = Math.round((segment.confidence || 0) * 100);
  const isLowConfidence = segment.confidence < 0.85;

  // Click row card to select this segment and seek to subtitle start time
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
    onSelect(segment);
  };

  const handleCardKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.target !== e.currentTarget) return;
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onSelect(segment);
    }
  };

  // ==========================================================================
  // COMPACT VIEW (Unselected Row - High Density Scannable View)
  // Displays: Line Number, Monospace Timecode, Original Text, Translated Text
  // ==========================================================================
  if (!isSelected) {
    return (
      <div
        role="button"
        tabIndex={0}
        onClick={handleCardClick}
        onKeyDown={handleCardKeyDown}
        aria-label={`Dòng phụ đề ${index + 1}, từ ${formatTimestamp(segment.start_time)} đến ${formatTimestamp(segment.end_time)}: ${localText}`}
        title="Nhấp để chọn sửa và chuyển video đến dòng phụ đề này"
        className={`group flex items-start gap-2.5 px-3 py-2 rounded-lg border transition-colors cursor-pointer select-text ${
          isPlayingNow
            ? 'bg-primary/12 border-primary/45 border-l-4 border-l-primary'
            : 'bg-base-100/80 border-base-content/15 hover:border-primary/35 hover:bg-base-100'
        }`}
      >
        {/* Left Column: Line Number + Monospace Timecode */}
        <div className="flex items-center gap-2 shrink-0 pt-0.5">
          <span
            className={`inline-flex items-center justify-center min-w-[28px] h-5 px-1.5 rounded text-[11px] font-mono tabular-nums font-bold ${
              isPlayingNow
                ? 'bg-primary text-primary-content'
                : 'bg-base-300 text-base-content/80 group-hover:bg-primary/20 group-hover:text-primary'
            }`}
          >
            {index + 1}
          </span>

          <span className="font-mono tabular-nums text-[11px] text-base-content/65 whitespace-nowrap">
            {formatTimestamp(segment.start_time)}
            <span className="text-base-content/35 mx-0.5">→</span>
            {formatTimestamp(segment.end_time)}
          </span>
        </div>

        {/* Center Column: Original Text & Translation (if available) */}
        <div className="flex-1 min-w-0">
          <div className="text-sm font-medium text-base-content leading-snug break-words">
            {localText || <span className="italic text-base-content/40">(Chưa có nội dung)</span>}
          </div>

          {localTranslated.trim().length > 0 && (
            <div className="text-xs text-primary dark:text-linen-400 leading-snug mt-0.5 break-words">
              {localTranslated}
            </div>
          )}
        </div>

        {/* Right Compact Badges (Only shown if low confidence or currently playing) */}
        {(isLowConfidence || isPlayingNow) && (
          <div className="flex items-center gap-1 shrink-0 pt-0.5">
            {isLowConfidence && (
              <span
                title={`Độ tin cậy thấp: ${confidencePercent}%`}
                className="badge badge-error badge-xs font-mono tabular-nums text-[10px] text-error-content py-2"
              >
                {confidencePercent}%
              </span>
            )}
          </div>
        )}
      </div>
    );
  }

  // ==========================================================================
  // EXPANDED EDIT VIEW (Selected Row - Stable Background, OCR Crop, Actions)
  // ==========================================================================
  const computedRows = Math.max(2, Math.min(5, Math.ceil((localText.length || 1) / 42)));

  return (
    <div
      onClick={handleCardClick}
      className="bg-base-100 border-2 border-primary border-l-4 border-l-primary rounded-xl p-3 shadow-xs transition-colors"
    >
      {/* Top Header: Index, Timestamps, Badges, and Row Action Buttons */}
      <div className="gap-2 mb-2.5 pb-2 border-b border-base-content/15">
      <div className='flex items-center justify-between'>
        <div className="flex items-center gap-1.5 flex-wrap">
          {/* Index Number + Active Label */}
          <span className="badge bg-primary text-primary-content border-none badge-sm font-bold font-mono tabular-nums">
            #{index + 1}
          </span>

          {/* Timestamp seeking button or editor */}
          {!isEditingTimes ? (
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => onSeek(segment.start_time)}
                title="Nhấp để phát video tại mốc thời gian này"
                className="btn btn-ghost btn-sm font-mono tabular-nums gap-1 text-primary hover:bg-primary/10 px-2 rounded-full"
              >
                <Clock className="w-3.5 h-3.5 text-primary" />
                <span>{formatTimestamp(segment.start_time)}</span>
                <span className="text-base-content/40">→</span>
                <span>{formatTimestamp(segment.end_time)}</span>
              </button>
              <button
                type="button"
                onClick={() => {
                  setIsEditingTimes(true);
                  onEditingFocusChange(true);
                }}
                title="Chỉnh sửa thời gian bắt đầu và kết thúc"
                className="btn btn-outline border-base-content/20 btn-sm px-2.5 text-xs text-base-content/75 hover:text-white hover:bg-primary rounded-full cursor-pointer"
              >
                Sửa giờ
              </button>
            </div>
          ) : (
            <div className="flex items-center gap-1">
              <input
                type="text"
                aria-label="Thời gian bắt đầu"
                value={startTimeStr}
                onFocus={() => onEditingFocusChange(true)}
                onChange={(e) => setStartTimeStr(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handleSaveTimes();
                  if (e.key === 'Escape') handleCancelTimes();
                }}
                className="input input-bordered input-sm font-mono tabular-nums w-24 bg-base-200  focus:border-primary rounded-lg text-xs"
              />
              <span className="text-base-content/40">→</span>
              <input
                type="text"
                aria-label="Thời gian kết thúc"
                value={endTimeStr}
                onFocus={() => onEditingFocusChange(true)}
                onChange={(e) => setEndTimeStr(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handleSaveTimes();
                  if (e.key === 'Escape') handleCancelTimes();
                }}
                className="input input-bordered input-sm font-mono tabular-nums w-24 bg-base-200  focus:border-primary rounded-lg text-xs"
              />
              <button
                type="button"
                onClick={handleSaveTimes}
                aria-label="Lưu thời gian"
                title="Lưu thời gian"
                className="btn btn-success btn-sm btn-circle cursor-pointer"
              >
                <Check className="w-3.5 h-3.5 text-success-content" />
              </button>
              <button
                type="button"
                onClick={handleCancelTimes}
                aria-label="Hủy sửa thời gian"
                title="Hủy sửa thời gian"
                className="btn btn-ghost btn-sm btn-circle cursor-pointer"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          )}

          {/* Source Badge */}
          <span
            className={`badge badge-xs font-mono font-semibold uppercase py-2 ${
              segment.source === 'ocr'
                ? 'badge-warning text-warning-content'
                : segment.source === 'stt'
                ? 'badge-info text-info-content'
                : 'badge-ghost text-base-content/75'
            }`}
          >
            {segment.source}
          </span>
        </div>

        {/* Action Controls (Merge, Split, Delete) - All icon buttons have aria-label & title */}
        <div className="flex items-center gap-1 shrink-0">
          {canMergeNext && (
            <button
              type="button"
              onClick={() => onMergeWithNext(segment.id)}
              aria-label="Gộp với câu tiếp theo"
              title="Gộp với câu tiếp theo"
              className="btn btn-ghost btn-sm btn-circle text-base-content/70 hover:text-primary hover:bg-base-200 cursor-pointer"
            >
              <Combine className="w-3.5 h-3.5" />
            </button>
          )}

          <button
            type="button"
            onClick={() => onOpenSplitDialog(segment)}
            aria-label="Tách câu phụ đề làm 2"
            title="Tách câu phụ đề làm 2"
            className="btn btn-ghost btn-sm btn-circle text-base-content/70 hover:text-primary hover:bg-base-200 cursor-pointer"
          >
            <Scissors className="w-3.5 h-3.5" />
          </button>

          <button
            type="button"
            onClick={() => onDelete(segment.id)}
            aria-label="Xóa câu phụ đề"
            title="Xóa câu phụ đề"
            className="btn btn-ghost btn-sm btn-circle text-base-content/60 hover:text-error hover:bg-error/10 cursor-pointer"
          >
            <Trash2 className="w-3.5 h-3.5" />
          </button>

          <button
            type="button"
            onClick={() => onClose()}
            aria-label="Thu gọn chi tiết câu phụ đề"
            title="Thu gọn / Đóng khung chỉnh sửa chi tiết"
            className="btn btn-outline  hover:border-primary hover:bg-base-200 btn-sm rounded-full gap-1 px-2.5 text-xs font-medium text-base-content/85 hover:text-primary cursor-pointer ml-1"
          >
            <ChevronUp className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
        {/* Confidence Badge */}
          <span
            title={`Độ tin cậy nhận diện: ${confidencePercent}%`}
            className={`badge badge-xs font-mono tabular-nums text-[10px] py-2 ${
              isLowConfidence
                ? 'badge-error text-error-content'
                : 'bg-base-200 text-base-content/70 border-base-content/15'
            }`}
          >
            {confidencePercent}%
          </span>
      </div>

      {/* Main Body: OCR Crop Thumbnail Preview + Text Input Area */}
      <div className="flex flex-col gap-2">
        {/* OCR Crop Image Preview (Shown when crop exists or segment is OCR) */}
        {segment.crop_image_url && !cropFailed ? (
          <div className="flex items-center gap-2.5 p-1.5 bg-base-200/70 rounded-lg border border-base-content/15">
            <button
              type="button"
              onClick={() => onOpenCropModal(segment)}
              aria-label="Xem ảnh crop OCR phóng to"
              title="Nhấp để phóng to ảnh crop OCR đối chiếu"
              className="relative group/crop flex-1 h-12 bg-dust_grey-100 rounded overflow-hidden cursor-pointer flex items-center justify-center"
            >
              <img
                src={segment.crop_image_url}
                alt={`Ảnh crop OCR dòng ${index + 1}`}
                onError={() => setCropFailed(true)}
                className="w-full h-full object-contain"
              />
              <div className="absolute inset-0 bg-black/45 opacity-0 group-hover/crop:opacity-100 flex items-center justify-center gap-1 transition-opacity text-white text-[11px] font-medium">
                <Maximize2 className="w-3.5 h-3.5" />
                <span>Phóng to ảnh OCR</span>
              </div>
            </button>
          </div>
        ) : segment.source === 'ocr' ? (
          <div
            title="Không có ảnh crop lưu cho dòng này"
            className="w-full h-9 bg-base-200/50 border border-dashed border-base-content/20 rounded-lg flex items-center justify-center gap-1.5 text-base-content/45 text-[11px]"
          >
            <ImageIcon className="w-3.5 h-3.5 text-base-content/40" />
            <span>Không có ảnh crop OCR</span>
          </div>
        ) : null}

        {/* Subtitle Text Content (Original and Translation) */}
        <div className="flex flex-col gap-2">
          {/* Original Text Input */}
          <div>
            <label
              htmlFor={`seg-orig-${segment.id}`}
              className="block text-[11px] font-semibold text-base-content/70 mb-1"
            >
              Văn bản gốc
            </label>
            <textarea
              id={`seg-orig-${segment.id}`}
              rows={computedRows}
              value={localText}
              onClick={(e) => e.stopPropagation()}
              onFocus={() => onEditingFocusChange(true)}
              onChange={handleTextChange}
              onBlur={handleBlurText}
              placeholder="Nhập nội dung phụ đề gốc..."
              className="textarea textarea-bordered textarea-sm w-full text-sm leading-relaxed bg-base-200/70 focus:bg-base-200  text-base-content focus:border-primary rounded-lg transition-colors font-sans"
            />
          </div>

          {/* Translated Text Input */}
          <div onClick={(e) => e.stopPropagation()}>
            <label
              htmlFor={`seg-trans-${segment.id}`}
              className="flex items-center gap-1 text-[11px] font-semibold text-base-content/70 mb-1"
            >
              <Languages className="w-3 h-3 text-secondary" />
              <span>Bản dịch</span>
            </label>
            <input
              id={`seg-trans-${segment.id}`}
              type="text"
              value={localTranslated}
              onClick={(e) => e.stopPropagation()}
              onFocus={() => onEditingFocusChange(true)}
              onChange={handleTranslatedChange}
              onBlur={handleBlurTranslated}
              placeholder="Nhập bản dịch phụ đề (tùy chọn)..."
              className="input input-bordered input-sm w-full text-xs bg-base-200/70 focus:bg-base-200  text-base-content focus:border-primary rounded-lg transition-colors font-sans"
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

  // Explicit selected segment ID for editing (does not close when video currentTime advances)
  const [selectedSegmentId, setSelectedSegmentId] = useState<string | null>(null);
  // Whether user is actively typing/focused inside an input or textarea in the selected row
  const [isEditingInput, setIsEditingInput] = useState<boolean>(false);

  // Clear selection if segments list is emptied or selected segment no longer exists
  useEffect(() => {
    if (selectedSegmentId && !segments.some((s) => s.id === selectedSegmentId)) {
      setSelectedSegmentId(null);
      setIsEditingInput(false);
    }
  }, [segments, selectedSegmentId]);

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

  // Count low confidence segments (< 85%)
  const lowConfidenceCount = useMemo(() => {
    return segments.filter((s) => s.confidence < 0.85).length;
  }, [segments]);

  // Identify currently playing segments at currentTime
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

  // Auto-scroll to active segment ONLY when user is NOT manually scrolling and NOT actively editing/typing
  useEffect(() => {
    if (
      !autoScroll ||
      isUserScrolling ||
      isEditingInput ||
      activeSegmentIndex === -1 ||
      !listContainerRef.current
    ) {
      return;
    }
    const container = listContainerRef.current;
    const activeElement = container.querySelector(`[data-segment-idx="${activeSegmentIndex}"]`);
    if (activeElement) {
      activeElement.scrollIntoView({
        behavior: 'smooth',
        block: 'nearest',
      });
    }
  }, [activeSegmentIndex, autoScroll, isUserScrolling, isEditingInput]);

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

  const handleSelectSegment = (seg: SubtitleSegment) => {
    setSelectedSegmentId(seg.id);
    setIsUserScrolling(false);
    onSeek(seg.start_time);
  };

  const handleSeekFromSegment = (time: number) => {
    setIsUserScrolling(false);
    onSeek(time);
  };

  const handleAddNewSegment = async () => {
    const created = await onAddSegment(currentTime);
    if (created && typeof created === 'object' && 'id' in created) {
      setSelectedSegmentId(created.id);
    }
  };

  return (
    <div
      className={`flex flex-col bg-base-200 rounded-xl border border-base-content/15 shadow-xs overflow-hidden h-full transition-smooth ${className}`}
    >
      {/* Toolbar Row 1: Editor Header & Grouped AI / Creation Actions */}
      <div className="flex flex-wrap items-center justify-between gap-2 px-3.5 py-2.5 bg-base-300/70 border-b border-base-content/15">
        {/* Title & Segment Count Badge */}
        <div className="flex items-center gap-2">
          <FileText className="w-4 h-4 text-primary" />
          <h2 className="font-bold text-base-content text-sm tracking-tight">Subtitle Editor</h2>
          <span
            title={`Hiển thị ${filteredSegments.length} trên tổng số ${segments.length} dòng phụ đề`}
            className="badge badge-sm bg-primary text-primary-content border-none font-mono tabular-nums"
          >
            {filteredSegments.length}/{segments.length}
          </span>
        </div>

        {/* Grouped Action Controls: [AI Group: AI Polish + Translate] | [Add Group: + Thêm] */}
        <div className="flex items-center gap-1.5 flex-wrap">
          {/* AI Processing Group */}
          <div className="flex items-center gap-1.5 flex-wrap">
            {onAIPolish && (
              <button
                type="button"
                onClick={() => onAIPolish(translateTarget, translationContext)}
                disabled={isProcessing || segments.length === 0}
                title="Dùng Gemini AI để trau chuốt và cải thiện bản dịch (giữ nguyên văn bản gốc)"
                className="btn btn-sm btn-outline  hover:border-primary hover:bg-base-300 hover:text-base-content text-base-content/90 gap-1.5 rounded-full cursor-pointer"
              >
                <Sparkles className="w-3.5 h-3.5 text-primary" />
                <span>AI Polish</span>
              </button>
            )}

            {onTranslate && (
              <div className="join border  rounded-full overflow-hidden">
                <select
                  aria-label="Ngữ cảnh dịch thuật"
                  value={translationContext}
                  onChange={handleTranslationContextChange}
                  disabled={isProcessing || segments.length === 0}
                  className="select select-bordered select-sm join-item bg-base-100 border-none text-xs cursor-pointer whitespace-nowrap"
                  title="Ngữ cảnh dịch: Đời thường (Mặc định), Cổ trang, Hiện đại"
                >
                  <option value="casual">Đời thường</option>
                  <option value="historical">Cổ trang</option>
                  <option value="modern">Hiện đại</option>
                </select>
                <select
                  aria-label="Ngôn ngữ dịch đích"
                  value={translateTarget}
                  onChange={handleTranslateTargetChange}
                  disabled={isProcessing || segments.length === 0}
                  className="select select-bordered select-sm join-item bg-base-100 border-none text-xs cursor-pointer whitespace-nowrap"
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
                  className="btn btn-sm btn-secondary join-item gap-1 border-none cursor-pointer font-semibold whitespace-nowrap"
                >
                  <Languages className="w-3.5 h-3.5" />
                  <span>Dịch</span>
                </button>
              </div>
            )}
          </div>

          <span className="text-base-content/25 hidden sm:inline">|</span>

          {/* Add New Segment Button */}
          <button
            type="button"
            onClick={handleAddNewSegment}
            disabled={isProcessing}
            title="Thêm một câu phụ đề mới tại mốc thời gian hiện tại"
            className="btn btn-sm btn-primary gap-1 rounded-full font-semibold cursor-pointer shadow-xs"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>Thêm</span>
          </button>
        </div>
      </div>

      {/* Toolbar Row 2: Search, Filter Group & Auto-scroll Toggle */}
      <div className="flex flex-wrap items-center gap-1.5 px-3.5 py-2 bg-base-300/40 border-b border-base-content/15 text-xs">
        {/* Search Input Group */}
        <div className="relative flex-1 min-w-[140px]">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-primary" />
          <input
            type="text"
            aria-label="Tìm kiếm phụ đề"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Tìm kiếm phụ đề..."
            className="input input-bordered input-sm w-full pl-8 pr-8 bg-base-100  focus:border-primary rounded-full text-xs"
          />
          {searchQuery && (
            <button
              type="button"
              onClick={() => setSearchQuery('')}
              aria-label="Xóa từ khóa tìm kiếm"
              title="Xóa từ khóa tìm kiếm"
              className="btn btn-ghost btn-xs btn-circle absolute right-1.5 top-1/2 -translate-y-1/2 text-base-content/50 hover:text-base-content cursor-pointer"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>

        {/* Filter Group: Source (DaisyUI select-sm) & Low Confidence */}
        <div className="flex items-center gap-1.5">
          <select
            aria-label="Lọc theo nguồn phụ đề"
            title="Lọc phụ đề theo nguồn tạo"
            value={sourceFilter}
            onChange={(e) => setSourceFilter(e.target.value as any)}
            className="select select-bordered select-sm text-xs bg-base-100  focus:border-primary rounded-full cursor-pointer"
          >
            <option value="all">Tất cả nguồn</option>
            <option value="stt">STT (Giọng nói)</option>
            <option value="ocr">OCR (Chữ viết)</option>
            <option value="manual">Thủ công</option>
          </select>

          <button
            type="button"
            aria-pressed={lowConfidenceOnly}
            onClick={() => setLowConfidenceOnly(!lowConfidenceOnly)}
            title="Chỉ hiển thị các dòng có độ tin cậy dưới 85% cần rà soát"
            className={`btn btn-sm gap-1 rounded-full cursor-pointer ${
              lowConfidenceOnly
                ? 'btn-error text-error-content'
                : 'btn-outline  hover:border-primary hover:bg-base-300 text-base-content/80 hover:text-base-content'
            }`}
          >
            <AlertCircle className="w-3.5 h-3.5" />
            <span>Độ tin cậy thấp</span>
            {lowConfidenceCount > 0 && (
              <span
                className={`badge badge-xs font-mono tabular-nums px-1.5 border-none ${
                  lowConfidenceOnly
                    ? 'bg-error-content text-error'
                    : 'bg-error/15 text-error'
                }`}
              >
                {lowConfidenceCount}
              </span>
            )}
          </button>
        </div>

        {/* Navigation / Auto-scroll Toggle */}
        <button
          type="button"
          aria-pressed={autoScroll}
          onClick={() => setAutoScroll(!autoScroll)}
          title="Tự động cuộn theo tiến trình video (Tạm dừng khi đang nhập liệu)"
          className={`btn btn-sm gap-1 rounded-full cursor-pointer ${
            autoScroll
              ? 'btn-primary btn-outline'
              : 'btn-outline border-base-content/20 text-base-content/55'
          }`}
        >
          <ArrowUpDown className="w-3.5 h-3.5" />
          <span>Tự cuộn</span>
        </button>
      </div>

      {/* Subtitles Scrollable List (High-Density Rows) */}
      <div
        ref={listContainerRef}
        onWheel={handleContainerManualScroll}
        onTouchMove={handleContainerManualScroll}
        className="flex-1 p-2.5 overflow-y-auto space-y-1.5 bg-base-200/40 relative"
      >
        {(isUserScrolling || isEditingInput) && autoScroll && (
          <div className="sticky top-1 z-30 flex justify-center pointer-events-none">
            <button
              type="button"
              onClick={() => {
                setIsUserScrolling(false);
                setIsEditingInput(false);
              }}
              className="btn btn-sm btn-neutral rounded-full shadow-md pointer-events-auto gap-1.5 text-xs animate-in fade-in cursor-pointer"
            >
              <ArrowUpDown className="w-3.5 h-3.5 text-accent" />
              <span>
                {isEditingInput
                  ? 'Đang giữ vị trí khi sửa phụ đề • Nhấp để tự cuộn tiếp'
                  : 'Tự cuộn đang tạm dừng • Nhấp để cuộn tiếp'}
              </span>
            </button>
          </div>
        )}

        {filteredSegments.length > 0 ? (
          filteredSegments.map((seg, idx) => {
            const originalIndex = segments.findIndex((s) => s.id === seg.id);
            const isPlayingNow = activeSegmentIds.has(seg.id);
            const isSelected = selectedSegmentId === seg.id;
            const canMergeNext = originalIndex < segments.length - 1;

            return (
              <div key={seg.id} data-segment-idx={originalIndex}>
                <SegmentRow
                  segment={seg}
                  index={originalIndex >= 0 ? originalIndex : idx}
                  isSelected={isSelected}
                  isPlayingNow={isPlayingNow}
                  canMergeNext={canMergeNext}
                  onSelect={handleSelectSegment}
                  onClose={() => {
                    setSelectedSegmentId(null);
                    setIsEditingInput(false);
                  }}
                  onSeek={handleSeekFromSegment}
                  onUpdate={onUpdateSegment}
                  onDelete={(id) => setPendingDeleteSegmentId(id)}
                  onMergeWithNext={handleMergeWithNext}
                  onOpenSplitDialog={handleOpenSplitDialog}
                  onOpenCropModal={(s) => setSelectedCropSegment(s)}
                  onEditingFocusChange={setIsEditingInput}
                />
              </div>
            );
          })
        ) : (
          <div className="flex flex-col items-center justify-center p-12 text-base-content/45">
            <FileText className="w-10 h-10 mb-2 text-base-content/30" />
            <p className="text-sm font-medium text-base-content/75">
              {segments.length === 0 ? 'Chưa có phụ đề nào' : 'Không tìm thấy phụ đề phù hợp'}
            </p>
            <p className="text-xs text-base-content/55 mt-1 text-center">
              {segments.length === 0
                ? 'Mở "Cấu hình OCR / Whisper" để trích xuất tự động, hoặc bấm "Thêm"'
                : 'Thử điều chỉnh từ khóa tìm kiếm hoặc bộ lọc'}
            </p>
          </div>
        )}
      </div>

      {/* Enlarged OCR Crop Modal (daisyUI modal) */}
      {selectedCropSegment && (
        <div className="modal modal-open bg-black/70 backdrop-blur-xs">
          <div className="modal-box max-w-lg p-0 overflow-hidden bg-base-200 border border-base-content/20 rounded-xl shadow-2xl">
            <div className="flex items-center justify-between px-5 py-3.5 bg-base-300/80 border-b border-base-content/15">
              <div className="flex items-center gap-2">
                <ImageIcon className="w-4 h-4 text-warning" />
                <h3 className="font-bold text-sm text-base-content">
                  Đối chiếu OCR Crop — Đoạn #{selectedCropSegment.index}
                </h3>
              </div>
              <button
                type="button"
                aria-label="Đóng cửa sổ đối chiếu OCR"
                title="Đóng"
                onClick={() => setSelectedCropSegment(null)}
                className="btn btn-ghost btn-sm btn-circle text-base-content/70 hover:text-base-content cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="p-5 flex flex-col gap-4">
              <div className="w-full bg-dust_grey-100 rounded-xl p-3 flex items-center justify-center min-h-[120px] max-h-[240px] overflow-hidden border border-base-content/20">
                <img
                  src={selectedCropSegment.crop_image_url || ''}
                  alt={`Ảnh crop phóng to đoạn ${selectedCropSegment.index}`}
                  className="max-w-full max-h-full object-contain"
                />
              </div>

              <div className="flex flex-col gap-2">
                <div className="flex items-center justify-between text-xs text-base-content/65 font-mono tabular-nums">
                  <span>
                    Thời gian: {formatTimestamp(selectedCropSegment.start_time)} → {formatTimestamp(selectedCropSegment.end_time)}
                  </span>
                  <span>Độ tin cậy: {Math.round(selectedCropSegment.confidence * 100)}%</span>
                </div>

                <div className="p-3 bg-base-100/80 rounded-xl border border-base-content/15 text-sm font-medium text-base-content">
                  {selectedCropSegment.text}
                </div>
              </div>
            </div>

            <div className="modal-action px-5 py-3 bg-base-300/50 m-0 border-t border-base-content/15">
              <button
                type="button"
                onClick={() => setSelectedCropSegment(null)}
                className="btn btn-sm btn-primary rounded-full font-semibold cursor-pointer"
              >
                Đóng
              </button>
            </div>
          </div>
          <div className="modal-backdrop bg-black/40" onClick={() => setSelectedCropSegment(null)} />
        </div>
      )}

      {/* Split Segment Dialog (daisyUI modal) */}
      {splitDialogSegment && (
        <div className="modal modal-open bg-black/70 backdrop-blur-xs">
          <div className="modal-box max-w-md p-0 overflow-hidden bg-base-200 border border-base-content/20 rounded-xl shadow-2xl">
            <div className="flex items-center justify-between px-5 py-3.5 bg-base-300/80 border-b border-base-content/15">
              <div className="flex items-center gap-2">
                <Scissors className="w-4 h-4 text-primary" />
                <h3 className="font-bold text-sm text-base-content">Tách đoạn phụ đề làm 2</h3>
              </div>
              <button
                type="button"
                aria-label="Đóng hộp thoại tách phụ đề"
                title="Đóng"
                onClick={() => setSplitDialogSegment(null)}
                className="btn btn-ghost btn-sm btn-circle text-base-content/70 hover:text-base-content cursor-pointer"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="p-5 flex flex-col gap-3.5 text-xs">
              <div>
                <label htmlFor="split-time-input" className="label label-text font-semibold py-1 text-base-content/85">
                  Mốc thời gian tách (giây)
                </label>
                <input
                  id="split-time-input"
                  type="number"
                  step="0.05"
                  min={splitDialogSegment.start_time + 0.05}
                  max={splitDialogSegment.end_time - 0.05}
                  value={splitTime}
                  onChange={(e) => setSplitTime(parseFloat(e.target.value))}
                  className="input input-bordered input-sm w-full font-mono tabular-nums bg-base-100 focus:border-primary rounded-lg text-xs"
                />
                <span className="text-[11px] text-base-content/60 mt-1 block font-mono tabular-nums">
                  Giới hạn: {splitDialogSegment.start_time.toFixed(2)}s đến {splitDialogSegment.end_time.toFixed(2)}s
                </span>
              </div>

              <div>
                <label htmlFor="split-first-text" className="label label-text font-semibold py-1 text-base-content/85">
                  Đoạn 1 ({splitDialogSegment.start_time.toFixed(2)}s - {splitTime.toFixed(2)}s)
                </label>
                <textarea
                  id="split-first-text"
                  rows={2}
                  value={splitFirstText}
                  onChange={(e) => setSplitFirstText(e.target.value)}
                  className="textarea textarea-bordered textarea-sm w-full bg-base-100 focus:border-primary rounded-lg text-xs font-sans"
                />
              </div>

              <div>
                <label htmlFor="split-second-text" className="label label-text font-semibold py-1 text-base-content/85">
                  Đoạn 2 ({splitTime.toFixed(2)}s - {splitDialogSegment.end_time.toFixed(2)}s)
                </label>
                <textarea
                  id="split-second-text"
                  rows={2}
                  value={splitSecondText}
                  onChange={(e) => setSplitSecondText(e.target.value)}
                  className="textarea textarea-bordered textarea-sm w-full bg-base-100 focus:border-primary rounded-lg text-xs font-sans"
                />
              </div>
            </div>

            <div className="modal-action px-5 py-3.5 bg-base-300/50 m-0 border-t border-base-content/15 gap-2">
              <button
                type="button"
                onClick={() => setSplitDialogSegment(null)}
                className="btn btn-sm btn-ghost rounded-full font-medium cursor-pointer"
              >
                Hủy
              </button>
              <button
                type="button"
                onClick={handleConfirmSplit}
                className="btn btn-sm btn-primary rounded-full font-semibold shadow-xs cursor-pointer"
              >
                Xác nhận tách
              </button>
            </div>
          </div>
          <div className="modal-backdrop bg-black/40" onClick={() => setSplitDialogSegment(null)} />
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
