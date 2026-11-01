import React, { useState, useEffect, useMemo, useRef } from 'react';
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
}

/**
 * Check if text contains Right-to-Left characters (Arabic, Hebrew, Persian, Urdu).
 */
export function isRTLText(text: string): boolean {
  if (!text) return false;
  // Unicode ranges for Hebrew, Arabic, Syriac, Thaana, Samaritan, Mandaic, Arabic Extended
  const rtlRegex = /[\u0591-\u07FF\uFB1D-\uFDFD\uFE70-\uFEFC]/;
  return rtlRegex.test(text);
}

/**
 * O(log N) Binary search algorithm to find active subtitle segments at `time`.
 * Supports multiple overlapping subtitle lines and variable segment durations.
 *
 * @param sortedSegments Subtitle segments sorted ascending by start_time
 * @param time Current playback time in seconds
 * @returns Array of currently active SubtitleSegments
 */
export function findActiveSegments(
  sortedSegments: SubtitleSegment[],
  time: number
): SubtitleSegment[] {
  if (sortedSegments.length === 0 || isNaN(time) || time < 0) {
    return [];
  }

  // Binary search for bisect_right on start_time:
  // Find highest index where sortedSegments[index].start_time <= time
  let low = 0;
  let high = sortedSegments.length - 1;
  let targetIndex = -1;

  while (low <= high) {
    const mid = (low + high) >> 1;
    if (sortedSegments[mid].start_time <= time) {
      targetIndex = mid;
      low = mid + 1; // Seek later potential candidates
    } else {
      high = mid - 1;
    }
  }

  if (targetIndex === -1) {
    return [];
  }

  const active: SubtitleSegment[] = [];
  // Scan backwards from targetIndex to collect all segments where end_time >= time.
  // Maximum lookback window of 30 seconds covers long speech lines while keeping O(K) bounded.
  const MAX_LOOKBACK_SECONDS = 30;

  for (let i = targetIndex; i >= 0; i--) {
    const seg = sortedSegments[i];
    if (time - seg.start_time > MAX_LOOKBACK_SECONDS) {
      break;
    }
    if (time >= seg.start_time && time <= seg.end_time) {
      active.unshift(seg); // Maintain chronological order
    }
  }

  return active;
}

export const SubtitleOverlay: React.FC<SubtitleOverlayProps> = ({
  videoRef,
  segments,
  renderRect,
  settings,
}) => {
  const [activeSegments, setActiveSegments] = useState<SubtitleSegment[]>([]);

  // Memoize sorted segments by start_time for binary search
  const sortedSegments = useMemo(() => {
    return [...segments].sort((a, b) => a.start_time - b.start_time);
  }, [segments]);

  // Keep a ref of sortedSegments for the animation frame loop to avoid recreation
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

      // Only update state if active segment IDs or text have changed
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

      // Chain next frame check if video is actively playing
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

    // Event listeners for seek, pause, timeupdate, playback rate change
    const handleImmediateUpdate = () => {
      checkSubtitlesAtCurrentTime();
    };

    video.addEventListener('timeupdate', handleImmediateUpdate);
    video.addEventListener('seeked', handleImmediateUpdate);
    video.addEventListener('pause', handleImmediateUpdate);
    video.addEventListener('play', handleImmediateUpdate);
    video.addEventListener('ratechange', handleImmediateUpdate);

    // Initial check on mount or video change
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

  // If subtitle overlay is disabled, or no active subtitles, or renderRect is invalid: render nothing
  if (
    !settings.enabled ||
    activeSegments.length === 0 ||
    renderRect.width <= 0 ||
    renderRect.height <= 0
  ) {
    return null;
  }

  // Positioning classes based on user settings
  let positionClass = 'justify-end pb-[5%]'; // Default 'bottom'
  if (settings.position === 'center') {
    positionClass = 'justify-center';
  } else if (settings.position === 'top') {
    positionClass = 'justify-start pt-[5%]';
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

  // Background style & text shadow for universal legibility
  let bgClasses = 'bg-black/65 backdrop-blur-xs rounded-lg px-3 py-1.5 shadow-lg';
  if (settings.background === 'none') {
    bgClasses = 'bg-transparent px-2 py-1';
  } else if (settings.background === 'solid') {
    bgClasses = 'bg-neutral-950/95 rounded-lg px-3.5 py-1.5 shadow-xl';
  }

  // High-contrast text stroke and drop shadow for crisp readability on any background
  const textShadowStyle: React.CSSProperties = {
    textShadow: `
      0 2px 4px rgba(0, 0, 0, 0.9),
      0 0 4px rgba(0, 0, 0, 0.9),
      0 0 1px #000000,
      -1px -1px 0 #000,
       1px -1px 0 #000,
      -1px  1px 0 #000,
       1px  1px 0 #000
    `,
  };

  return (
    <div
      className={`absolute inset-0 pointer-events-none select-none z-20 flex flex-col items-center ${positionClass}`}
      style={{
        left: `${renderRect.left}px`,
        top: `${renderRect.top}px`,
        width: `${renderRect.width}px`,
        height: `${renderRect.height}px`,
        containerType: 'inline-size',
      }}
    >
      <div className="flex flex-col items-center gap-2 max-w-[92%] px-2 text-center pointer-events-none">
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
              className={`${bgClasses} transition-all duration-100 flex flex-col items-center max-w-full`}
              style={fontSizeStyle}
            >
              {/* Original Subtitle Line */}
              {showOriginal && (
                <div
                  dir={isOriginalRTL ? 'rtl' : 'ltr'}
                  style={textShadowStyle}
                  className="font-medium text-white break-words whitespace-pre-wrap leading-snug tracking-wide font-sans"
                >
                  {segment.text}
                </div>
              )}

              {/* Translated Subtitle Line (Distinctive warm golden-sand tint for bilingual clarity) */}
              {showTranslation && (
                <div
                  dir={isTranslatedRTL ? 'rtl' : 'ltr'}
                  style={textShadowStyle}
                  className={`font-semibold text-powder-petal-300 break-words whitespace-pre-wrap leading-snug tracking-wide font-sans ${
                    showOriginal ? 'mt-0.5 text-[0.92em]' : 'text-white'
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
  );
};

export default SubtitleOverlay;
