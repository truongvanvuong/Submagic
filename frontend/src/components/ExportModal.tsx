import React, { useState, useEffect, useMemo } from 'react';
import {
  X,
  Download,
  Copy,
  Check,
  FileCode,
  FileText,
  Languages,
  AlertCircle,
  Loader2,
} from 'lucide-react';
import type {
  BilingualLayout,
  ExportFormat,
  ExportRequest,
  ProjectDetail,
} from '../types/subtitle';
import { exportSubtitles } from '../services/api';

export interface ExportModalProps {
  isOpen: boolean;
  onClose: () => void;
  project: ProjectDetail | null;
  onExportSuccess?: (filename: string) => void;
}

/**
 * Format timestamp in seconds to SRT timecode: 00:00:00,000
 */
function formatSrtTime(seconds: number): string {
  const s = Math.max(0, seconds);
  const hrs = Math.floor(s / 3600);
  const mins = Math.floor((s % 3600) / 60);
  const secs = Math.floor(s % 60);
  const ms = Math.floor((s - Math.floor(s)) * 1000);

  return `${String(hrs).padStart(2, '0')}:${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')},${String(ms).padStart(3, '0')}`;
}

/**
 * Format timestamp in seconds to VTT timecode: 00:00:00.000
 */
function formatVttTime(seconds: number): string {
  const s = Math.max(0, seconds);
  const hrs = Math.floor(s / 3600);
  const mins = Math.floor((s % 3600) / 60);
  const secs = Math.floor(s % 60);
  const ms = Math.floor((s - Math.floor(s)) * 1000);

  return `${String(hrs).padStart(2, '0')}:${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}.${String(ms).padStart(3, '0')}`;
}

export const ExportModal: React.FC<ExportModalProps> = ({
  isOpen,
  onClose,
  project,
  onExportSuccess,
}) => {
  const [format, setFormat] = useState<ExportFormat>('srt');
  const [bilingualLayout, setBilingualLayout] = useState<BilingualLayout>('none');
  const [includeTimestampsInTxt, setIncludeTimestampsInTxt] = useState<boolean>(true);
  const [isExporting, setIsExporting] = useState<boolean>(false);
  const [copied, setCopied] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Close on Escape key press
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  // Check if project has translations
  const hasTranslations = useMemo(() => {
    if (!project?.segments) return false;
    return project.segments.some(
      (s) => s.translated_text && s.translated_text.trim().length > 0
    );
  }, [project?.segments]);

  // Generate client-side preview of the formatted subtitle
  const previewText = useMemo(() => {
    if (!project || !project.segments || project.segments.length === 0) {
      return '(Chưa có câu phụ đề nào để xuất)';
    }

    const segments = project.segments.slice(0, 10);
    const lines: string[] = [];

    if (format === 'vtt') {
      lines.push('WEBVTT\n');
    }

    segments.forEach((seg, i) => {
      const orig = seg.text || '';
      const trans = seg.translated_text || '';

      let textToDisplay = orig;
      if (bilingualLayout === 'original_top') {
        textToDisplay = trans ? `${orig}\n${trans}` : orig;
      } else if (bilingualLayout === 'translated_top') {
        textToDisplay = trans ? `${trans}\n${orig}` : orig;
      } else if (bilingualLayout === 'translated_only') {
        textToDisplay = trans || orig;
      }

      if (format === 'srt') {
        lines.push(`${i + 1}`);
        lines.push(`${formatSrtTime(seg.start_time)} --> ${formatSrtTime(seg.end_time)}`);
        lines.push(textToDisplay);
        lines.push('');
      } else if (format === 'vtt') {
        lines.push(`${i + 1}`);
        lines.push(`${formatVttTime(seg.start_time)} --> ${formatVttTime(seg.end_time)}`);
        lines.push(textToDisplay);
        lines.push('');
      } else {
        if (includeTimestampsInTxt) {
          lines.push(`[${formatVttTime(seg.start_time)} --> ${formatVttTime(seg.end_time)}] ${textToDisplay.replace(/\n/g, ' ')}`);
        } else {
          lines.push(textToDisplay.replace(/\n/g, ' '));
        }
      }
    });

    if (project.segments.length > 10) {
      lines.push(`... (+${project.segments.length - 10} câu phụ đề nữa)`);
    }

    return lines.join('\n');
  }, [project, format, bilingualLayout, includeTimestampsInTxt]);

  if (!isOpen) return null;

  // Trigger export download via API
  const handleDownload = async () => {
    if (!project) return;
    setIsExporting(true);
    setErrorMsg(null);

    const exportReq: ExportRequest = {
      format,
      bilingual_layout: bilingualLayout,
      include_timestamps_in_txt: includeTimestampsInTxt,
    };

    try {
      const res = await exportSubtitles(project.id, exportReq);
      const url = window.URL.createObjectURL(res.blob);
      const a = document.createElement('a');
      a.href = url;
      const cleanProjectName = project.name.replace(/[^a-zA-Z0-9_\u00C0-\u024F\u1E00-\u1EFF-]/g, '_');
      const filename = res.filename || `${cleanProjectName}.${format}`;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      if (onExportSuccess) {
        onExportSuccess(filename);
      }
      onClose();
    } catch (err: any) {
      console.error('Export download failed:', err);
      setErrorMsg(err.message || 'Xuất tệp thất bại. Vui lòng thử lại.');
    } finally {
      setIsExporting(false);
    }
  };

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(previewText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // ignore
    }
  };

  return (
    <div className="modal modal-open bg-blue_slate-100/70 backdrop-blur-xs">
      <div className="modal-box max-w-2xl p-0 overflow-hidden bg-base-200 border border-blue_slate-500/30 rounded-2xl shadow-2xl">
        {/* Modal Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-blue_slate-500/20 bg-base-300/80">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-glaucous-500/15 text-glaucous-600 dark:text-pale_sky-400 rounded-xl border border-blue_slate-500/30 shadow-xs">
              <Download className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-base font-bold text-base-content tracking-tight">
                Xuất Tệp Phụ Đề
              </h3>
              <p className="text-xs text-blue_slate-600 dark:text-blue_slate-800 mt-0.5">
                {project?.name || 'Dự án'} • {project?.segments?.length || 0} câu phụ đề
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="btn btn-ghost btn-xs btn-square text-base-content/70 hover:text-base-content rounded-lg"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Modal Content */}
        <div className="p-6 overflow-y-auto space-y-5 text-xs max-h-[70vh]">
          {/* Format Selection Cards */}
          <div>
            <label className="block text-xs font-bold text-base-content mb-2 tracking-tight">
              1. Chọn định dạng tệp xuất
            </label>
            <div className="grid grid-cols-3 gap-3">
              {/* SRT */}
              <div
                onClick={() => setFormat('srt')}
                className={`card card-bordered p-3.5 cursor-pointer rounded-xl transition-smooth ${
                  format === 'srt'
                    ? 'border-glaucous-500 bg-glaucous-500/10 ring-1 ring-glaucous-400/50 shadow-sm'
                    : 'bg-base-100 border-blue_slate-500/25 hover:border-glaucous-500/40 hover:bg-base-100'
                }`}
              >
                <div className="flex items-center justify-between w-full mb-1">
                  <span className="font-bold text-sm text-base-content font-mono">.SRT</span>
                  <FileCode className={`w-4 h-4 ${format === 'srt' ? 'text-glaucous-600 dark:text-pale_sky-400' : 'text-base-content/40'}`} />
                </div>
                <p className="text-[11px] text-base-content/60 leading-tight">
                  Chuẩn SubRip cho Premiere, VLC, CapCut
                </p>
              </div>

              {/* VTT */}
              <div
                onClick={() => setFormat('vtt')}
                className={`card card-bordered p-3.5 cursor-pointer rounded-xl transition-smooth ${
                  format === 'vtt'
                    ? 'border-glaucous-500 bg-glaucous-500/10 ring-1 ring-glaucous-400/50 shadow-sm'
                    : 'bg-base-100 border-blue_slate-500/25 hover:border-glaucous-500/40 hover:bg-base-100'
                }`}
              >
                <div className="flex items-center justify-between w-full mb-1">
                  <span className="font-bold text-sm text-base-content font-mono">.VTT</span>
                  <FileCode className={`w-4 h-4 ${format === 'vtt' ? 'text-glaucous-600 dark:text-pale_sky-400' : 'text-base-content/40'}`} />
                </div>
                <p className="text-[11px] text-base-content/60 leading-tight">
                  Chuẩn WebVTT cho trình phát Web HTML5
                </p>
              </div>

              {/* TXT */}
              <div
                onClick={() => setFormat('txt')}
                className={`card card-bordered p-3.5 cursor-pointer rounded-xl transition-smooth ${
                  format === 'txt'
                    ? 'border-glaucous-500 bg-glaucous-500/10 ring-1 ring-glaucous-400/50 shadow-sm'
                    : 'bg-base-100 border-blue_slate-500/25 hover:border-glaucous-500/40 hover:bg-base-100'
                }`}
              >
                <div className="flex items-center justify-between w-full mb-1">
                  <span className="font-bold text-sm text-base-content font-mono">.TXT</span>
                  <FileText className={`w-4 h-4 ${format === 'txt' ? 'text-glaucous-600 dark:text-pale_sky-400' : 'text-base-content/40'}`} />
                </div>
                <p className="text-[11px] text-base-content/60 leading-tight">
                  Văn bản thuần làm kịch bản hoặc tóm tắt
                </p>
              </div>
            </div>
          </div>

          {/* Bilingual Layout Options */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-bold text-base-content flex items-center gap-1.5 tracking-tight">
                <Languages className="w-4 h-4 text-primary" />
                2. Bố cục Phụ đề Song ngữ
              </label>

              {!hasTranslations && (
                <span className="badge badge-warning text-warning-content badge-sm gap-1 text-[11px]">
                  <AlertCircle className="w-3 h-3" /> Chưa có bản dịch AI
                </span>
              )}
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              <label
                className={`card card-bordered p-3 cursor-pointer flex-row items-start gap-2.5 rounded-xl transition-smooth ${
                  bilingualLayout === 'none'
                    ? 'border-glaucous-500 bg-glaucous-500/10'
                    : 'bg-base-100 border-blue_slate-500/25 hover:bg-base-100'
                }`}
              >
                <input
                  type="radio"
                  name="bilingual"
                  checked={bilingualLayout === 'none'}
                  onChange={() => setBilingualLayout('none')}
                  className="radio radio-primary radio-xs mt-0.5"
                />
                <div>
                  <span className="font-semibold text-base-content">Đơn ngữ (Chỉ chữ gốc)</span>
                  <p className="text-[11px] text-base-content/60 mt-0.5">Xuất duy nhất dòng chữ gốc nhận diện được.</p>
                </div>
              </label>

              <label
                className={`card card-bordered p-3 cursor-pointer flex-row items-start gap-2.5 rounded-xl transition-smooth ${
                  bilingualLayout === 'original_top'
                    ? 'border-glaucous-500 bg-glaucous-500/10'
                    : 'bg-base-100 border-blue_slate-500/25 hover:bg-base-100'
                }`}
              >
                <input
                  type="radio"
                  name="bilingual"
                  checked={bilingualLayout === 'original_top'}
                  onChange={() => setBilingualLayout('original_top')}
                  className="radio radio-primary radio-xs mt-0.5"
                />
                <div>
                  <span className="font-semibold text-base-content">Song ngữ (Gốc trên, Dịch dưới)</span>
                  <p className="text-[11px] text-base-content/60 mt-0.5">Dòng 1: Tiếng gốc • Dòng 2: Bản dịch</p>
                </div>
              </label>

              <label
                className={`card card-bordered p-3 cursor-pointer flex-row items-start gap-2.5 rounded-xl transition-smooth ${
                  bilingualLayout === 'translated_top'
                    ? 'border-glaucous-500 bg-glaucous-500/10'
                    : 'bg-base-100 border-blue_slate-500/25 hover:bg-base-100'
                }`}
              >
                <input
                  type="radio"
                  name="bilingual"
                  checked={bilingualLayout === 'translated_top'}
                  onChange={() => setBilingualLayout('translated_top')}
                  className="radio radio-primary radio-xs mt-0.5"
                />
                <div>
                  <span className="font-semibold text-base-content">Song ngữ (Dịch trên, Gốc dưới)</span>
                  <p className="text-[11px] text-base-content/60 mt-0.5">Dòng 1: Bản dịch • Dòng 2: Tiếng gốc</p>
                </div>
              </label>

              <label
                className={`card card-bordered p-3 cursor-pointer flex-row items-start gap-2.5 rounded-xl transition-smooth ${
                  bilingualLayout === 'translated_only'
                    ? 'border-glaucous-500 bg-glaucous-500/10'
                    : 'bg-base-100 border-blue_slate-500/25 hover:bg-base-100'
                }`}
              >
                <input
                  type="radio"
                  name="bilingual"
                  checked={bilingualLayout === 'translated_only'}
                  onChange={() => setBilingualLayout('translated_only')}
                  className="radio radio-primary radio-xs mt-0.5"
                />
                <div>
                  <span className="font-semibold text-base-content">Chỉ xuất bản dịch (Translated Only)</span>
                  <p className="text-[11px] text-base-content/60 mt-0.5">Thay thế toàn bộ bằng ngôn ngữ đã dịch.</p>
                </div>
              </label>
            </div>
          </div>

          {/* TXT Format Extra Checkbox */}
          {format === 'txt' && (
            <div className="flex items-center gap-2.5 p-3 bg-base-100 rounded-xl border border-blue_slate-500/25">
              <input
                type="checkbox"
                id="txt-timestamps"
                checked={includeTimestampsInTxt}
                onChange={(e) => setIncludeTimestampsInTxt(e.target.checked)}
                className="checkbox checkbox-primary checkbox-xs rounded"
              />
              <label htmlFor="txt-timestamps" className="text-xs text-base-content/85 select-none cursor-pointer">
                Đính kèm mốc thời gian <span className="font-mono text-glaucous-600 dark:text-pale_sky-400">[00:00:01.000 --&gt; 00:00:04.000]</span> ở đầu mỗi câu
              </label>
            </div>
          )}

          {/* Live Preview Box */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-xs font-bold text-base-content tracking-tight">
                3. Xem trước nội dung ({format.toUpperCase()})
              </label>
              <button
                type="button"
                onClick={handleCopy}
                className="btn btn-ghost btn-xs gap-1 text-[11px] text-base-content/75 hover:text-base-content rounded-md"
              >
                {copied ? (
                  <Check className="w-3.5 h-3.5 text-success" />
                ) : (
                  <Copy className="w-3.5 h-3.5" />
                )}
                <span>{copied ? 'Đã sao chép' : 'Sao chép xem trước'}</span>
              </button>
            </div>

            <div className="bg-blue_slate-100 text-ghost_white-DEFAULT rounded-xl p-3.5 font-mono text-[11px] leading-relaxed max-h-40 overflow-y-auto border border-blue_slate-500/30 whitespace-pre shadow-inner">
              {previewText}
            </div>
          </div>

          {/* Error Message */}
          {errorMsg && (
            <div className="alert alert-error text-xs py-2 rounded-xl">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{errorMsg}</span>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="modal-action px-6 py-4 m-0 border-t border-blue_slate-500/20 bg-base-300/60 gap-2">
          <button
            type="button"
            onClick={onClose}
            disabled={isExporting}
            className="btn btn-sm btn-ghost rounded-lg font-medium"
          >
            Đóng
          </button>

          <button
            type="button"
            onClick={handleDownload}
            disabled={isExporting || !project || !project.segments || project.segments.length === 0}
            className="btn btn-sm btn-primary gap-1.5 rounded-lg font-semibold shadow-md cursor-pointer disabled:cursor-not-allowed"
          >
            {isExporting ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>Đang xuất file...</span>
              </>
            ) : (
              <>
                <Download className="w-3.5 h-3.5" />
                <span>Tải tệp .{format.toUpperCase()} về máy</span>
              </>
            )}
          </button>
        </div>
      </div>
      <div className="modal-backdrop bg-blue_slate-100/50" onClick={onClose} />
    </div>
  );
};

export default ExportModal;
