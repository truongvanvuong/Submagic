import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { Move } from 'lucide-react';
import type { SubtitleSegment, SubtitleOverlaySettings } from '../types/subtitle';

export interface SubtitleOverlayProps {
  videoRef: React.RefObject<HTMLVideoElement>;
  segments: SubtitleSegment[];
  renderRect: {
    left: number;
    top: number;
    width: number;
    height: number;
  };
  settings: SubtitleOverlaySettings;
  onPositionChange?: (pos: { x: number; y: number }) => void;
}

/**
 * Check if text contains Right-to-Left characters (Arabic, Hebrew, Persian, Urdu).
 */
export function isRTLText(text: string): boolean {
  if (!text) return false;
  const rtlRegex = /[\u0591-\u07FF\uFB1D-\uFDFD\uFE70-\uFEFC]/;
  return rtlRegex.test(text);
}

/**
 * O(log N) Binary search algorithm to find active subtitle segments at `time`.
 */
export function findActiveSegments(
  sortedSegments: SubtitleSegment[],
  time: number
): SubtitleSegment[] {
  if (sortedSegments.length === 0 || isNaN(time) || time < 0) {
    return [];
  }

  let low = 0;
  let high = sortedSegments.length - 1;
  let targetIndex = -1;

  while (low <= high) {
    const mid = (low + high) >> 1;
    if (sortedSegments[mid].start_time <= time) {
      targetIndex = mid;
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }

  if (targetIndex === -1) {
    return [];
  }

  const active: SubtitleSegment[] = [];
  const MAX_LOOKBACK_SECONDS = 30;

  for (let i = targetIndex; i >= 0; i--) {
    const seg = sortedSegments[i];
    if (time - seg.start_time > MAX_LOOKBACK_SECONDS) {
      break;
    }
    if (time >= seg.start_time && time <= seg.end_time) {
      active.unshift(seg);
    }
  }

  return active;
}

export const SubtitleOverlay: React.FC<SubtitleOverlayProps> = ({
  videoRef,
  segments,
  renderRect,
  settings,
  onPositionChange,
}) => {
  const [activeSegments, setActiveSegments] = useState<SubtitleSegment[]>([]);

  // Dragging state for interactive positioning
  const [isDragging, setIsDragging] = useState(false);
  const dragStartRef = useRef<{ startMouseX: number; startMouseY: number; startPosX: number; startPosY: number } | null>(null);

  // Default position: Center X (0.5), Bottom Y (0.88)
  const currentPos = settings.positionOffset || { x: 0.5, y: 0.88 };

  // Memoize sorted segments by start_time for binary search
  const sortedSegments = useMemo(() => {
    return [...segments].sort((a, b) => a.start_time - b.start_time);
  }, [segments]);

  const sortedSegmentsRef = useRef<SubtitleSegment[]>(sortedSegments);
  useEffect(() => {
    sortedSegmentsRef.current = sortedSegments;
  }, [sortedSegments]);

  // Frame-accurate synchronization via requestVideoFrameCallback or requestAnimationFrame
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    let isDestroyed = false;
    let videoCallbackId: number | null = null;
    let animFrameId: number | null = null;

    const checkSubtitlesAtCurrentTime = () => {
      if (isDestroyed) return;
      const t = video.currentTime;
      const candidates = findActiveSegments(sortedSegmentsRef.current, t);

      setActiveSegments((prev) => {
        if (prev.length !== candidates.length) return candidates;
        const isIdentical = prev.every((p, idx) => {
          const c = candidates[idx];
          return (
            p.id === c?.id &&
            p.text === c?.text &&
            p.translated_text === c?.translated_text
          );
        });
        return isIdentical ? prev : candidates;
      });

      if (!video.paused && !video.ended) {
        if ('requestVideoFrameCallback' in video) {
          videoCallbackId = (video as any).requestVideoFrameCallback(
            checkSubtitlesAtCurrentTime
          );
        } else {
          animFrameId = requestAnimationFrame(checkSubtitlesAtCurrentTime);
        }
      }
    };

    const handleImmediateUpdate = () => {
      checkSubtitlesAtCurrentTime();
    };

    video.addEventListener('timeupdate', handleImmediateUpdate);
    video.addEventListener('seeked', handleImmediateUpdate);
    video.addEventListener('pause', handleImmediateUpdate);
    video.addEventListener('play', handleImmediateUpdate);
    video.addEventListener('ratechange', handleImmediateUpdate);

    checkSubtitlesAtCurrentTime();

    return () => {
      isDestroyed = true;
      if (videoCallbackId !== null && 'cancelVideoFrameCallback' in video) {
        (video as any).cancelVideoFrameCallback(videoCallbackId);
      }
      if (animFrameId !== null) {
        cancelAnimationFrame(animFrameId);
      }
      video.removeEventListener('timeupdate', handleImmediateUpdate);
      video.removeEventListener('seeked', handleImmediateUpdate);
      video.removeEventListener('pause', handleImmediateUpdate);
      video.removeEventListener('play', handleImmediateUpdate);
      video.removeEventListener('ratechange', handleImmediateUpdate);
    };
  }, [videoRef]);

  // Drag handlers to reposition subtitle box on video
  const handleMouseDown = useCallback((e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    if (renderRect.width <= 0 || renderRect.height <= 0) return;

    setIsDragging(true);
    dragStartRef.current = {
      startMouseX: e.clientX,
      startMouseY: e.clientY,
      startPosX: currentPos.x,
      startPosY: currentPos.y,
    };

    const handleMouseMove = (moveEvent: MouseEvent) => {
      if (!dragStartRef.current || renderRect.width <= 0 || renderRect.height <= 0) return;

      const deltaPixelX = moveEvent.clientX - dragStartRef.current.startMouseX;
      const deltaPixelY = moveEvent.clientY - dragStartRef.current.startMouseY;

      const deltaNormX = deltaPixelX / renderRect.width;
      const deltaNormY = deltaPixelY / renderRect.height;

      // Clamp between 0.08 and 0.92 so box stays inside video viewport
      const nextX = Math.max(0.08, Math.min(0.92, dragStartRef.current.startPosX + deltaNormX));
      const nextY = Math.max(0.05, Math.min(0.95, dragStartRef.current.startPosY + deltaNormY));

      if (onPositionChange) {
        onPositionChange({
          x: Number(nextX.toFixed(4)),
          y: Number(nextY.toFixed(4)),
        });
      }
    };

    const handleMouseUp = () => {
      setIsDragging(false);
      dragStartRef.current = null;
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
  }, [renderRect, currentPos, onPositionChange]);

  if (
    !settings.enabled ||
    activeSegments.length === 0 ||
    renderRect.width <= 0 ||
    renderRect.height <= 0
  ) {
    return null;
  }

  // Responsive font size styling via container query units (cqw) or relative clamps
  let fontSizeStyle: React.CSSProperties = {};
  if (settings.fontSize === 'small') {
    fontSizeStyle = { fontSize: 'clamp(12px, 2.0cqw, 20px)' };
  } else if (settings.fontSize === 'large') {
    fontSizeStyle = { fontSize: 'clamp(18px, 3.8cqw, 42px)' };
  } else {
    // Medium (Default)
    fontSizeStyle = { fontSize: 'clamp(14px, 2.8cqw, 30px)' };
  }

  // Default design: Black background with white text as requested
  const bgClasses = 'bg-black/85 backdrop-blur-xs rounded-xl px-4 py-2 shadow-2xl border border-white/15';

  // High-contrast text drop shadow for crisp readability on any background
  const textShadowStyle: React.CSSProperties = {
    textShadow: `
      0 2px 4px rgba(0, 0, 0, 0.95),
      0 0 2px #000000,
      -1px -1px 0 #000,
       1px -1px 0 #000,
      -1px  1px 0 #000,
       1px  1px 0 #000
    `,
  };

  return (
    <div
      className="absolute inset-0 pointer-events-none select-none z-20"
      style={{
        left: `${renderRect.left}px`,
        top: `${renderRect.top}px`,
        width: `${renderRect.width}px`,
        height: `${renderRect.height}px`,
        containerType: 'inline-size',
      }}
    >
      {/* Draggable Subtitle Box Container */}
      <div
        onMouseDown={handleMouseDown}
        title="Nhấp giữ và kéo để di chuyển vị trí phụ đề"
        className={`absolute pointer-events-auto select-none transition-shadow ${
          isDragging
            ? 'cursor-grabbing ring-2 ring-primary ring-offset-2 ring-offset-black/50 shadow-2xl scale-[1.01]'
            : 'cursor-grab hover:ring-1 hover:ring-white/40'
        }`}
        style={{
          left: `${currentPos.x * 100}%`,
          top: `${currentPos.y * 100}%`,
          transform: 'translate(-50%, -50%)',
          maxWidth: '92%',
        }}
      >
        <div className="relative group/box">
          {/* Subtle drag handle on hover */}
          <div className="absolute -top-3.5 left-1/2 -translate-x-1/2 opacity-0 group-hover/box:opacity-100 transition-opacity bg-black/80 text-white/80 rounded-full px-2 py-0.5 text-[9px] font-mono flex items-center gap-1 shadow pointer-events-none whitespace-nowrap">
            <Move className="w-2.5 h-2.5" />
            <span>Kéo để di chuyển</span>
          </div>

          <div className="flex flex-col items-center gap-1.5 text-center">
            {activeSegments.map((segment) => {
              const hasTranslation = Boolean(segment.translated_text?.trim());
              const showOriginal =
                settings.languageMode === 'original' ||
                settings.languageMode === 'bilingual' ||
                !hasTranslation;
              const showTranslation =
                hasTranslation &&
                (settings.languageMode === 'translated' ||
                  settings.languageMode === 'bilingual');

              const isOriginalRTL = isRTLText(segment.text);
              const isTranslatedRTL = segment.translated_text
                ? isRTLText(segment.translated_text)
                : false;

              return (
                <div
                  key={segment.id}
                  className={`${bgClasses} flex flex-col items-center max-w-full font-sans`}
                  style={fontSizeStyle}
                >
                  {/* Original Subtitle Line: White Text */}
                  {showOriginal && (
                    <div
                      dir={isOriginalRTL ? 'rtl' : 'ltr'}
                      style={textShadowStyle}
                      className="font-medium text-white break-words whitespace-pre-wrap leading-snug tracking-wide"
                    >
                      {segment.text}
                    </div>
                  )}

                  {/* Translated Subtitle Line (if bilingual): Subtle separator with crisp text */}
                  {showTranslation && (
                    <div
                      dir={isTranslatedRTL ? 'rtl' : 'ltr'}
                      style={textShadowStyle}
                      className={`font-medium text-gray-200 break-words whitespace-pre-wrap leading-snug tracking-wide ${
                        showOriginal ? 'text-[0.92em] mt-1 pt-1 border-t border-white/20' : 'text-white'
                      }`}
                    >
                      {segment.translated_text}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};

export default SubtitleOverlay;
