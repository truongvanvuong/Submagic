import React, { useState, useEffect, useRef } from 'react';
import {
  Terminal,
  ChevronDown,
  ChevronUp,
  Trash2,
  Copy,
  Check,
  Activity,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Clock,
  Sliders,
} from 'lucide-react';
import type { JobStatus } from '../types/subtitle';

export interface LogEntry {
  id?: string;
  timestamp: string;
  message: string;
  type?: 'info' | 'success' | 'warning' | 'error';
}

export interface ProgressBarProps {
  isProcessing: boolean;
  progress: number; // 0 - 100
  status: JobStatus;
  statusText?: string;
  logs: LogEntry[];
  onClearLogs?: () => void;
  isConfigOpen?: boolean;
  onToggleConfig?: () => void;
  onCancelJob?: () => void;
  className?: string;
}

export const ProgressBar: React.FC<ProgressBarProps> = ({
  isProcessing,
  progress,
  status,
  statusText,
  logs,
  onClearLogs,
  isConfigOpen = false,
  onToggleConfig,
  onCancelJob,
  className = '',
}) => {
  // Collapsible log stream (closed by default so workspace stays compact, openable anytime)
  const [isTerminalExpanded, setIsTerminalExpanded] = useState<boolean>(false);
  const [copied, setCopied] = useState<boolean>(false);
  const terminalEndRef = useRef<HTMLDivElement | null>(null);

  const clampedProgress = Math.max(0, Math.min(100, Math.round(progress)));

  useEffect(() => {
    if (isTerminalExpanded && terminalEndRef.current) {
      terminalEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs.length, isTerminalExpanded]);

  const handleCopyLogs = async () => {
    if (logs.length === 0) return;
    const text = logs
      .map((l) => `[${l.timestamp}] [${(l.type || 'info').toUpperCase()}] ${l.message}`)
      .join('\n');
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // ignore
    }
  };

  const getStatusBadge = () => {
    switch (status) {
      case 'running':
        return (
          <span className="badge badge-info text-info-content badge-sm gap-1 font-semibold whitespace-nowrap shrink-0 animate-pulse">
            <span className="w-1.5 h-1.5 rounded-full bg-accent animate-ping shrink-0" />
            <span>Đang xử lý ({clampedProgress}%)</span>
          </span>
        );
      case 'completed':
        return (
          <span className="badge badge-success text-success-content badge-sm gap-1 font-semibold whitespace-nowrap shrink-0">
            <CheckCircle2 className="w-3 h-3 shrink-0" />
            <span>Hoàn tất 100%</span>
          </span>
        );
      case 'failed':
        return (
          <span className="badge badge-error text-error-content badge-sm gap-1 font-semibold whitespace-nowrap shrink-0">
            <XCircle className="w-3 h-3 shrink-0" />
            <span>Thất bại</span>
          </span>
        );
      case 'cancelled':
        return (
          <span className="badge badge-warning text-warning-content badge-sm gap-1 font-semibold whitespace-nowrap shrink-0">
            <AlertTriangle className="w-3 h-3 shrink-0" />
            <span>Đã hủy</span>
          </span>
        );
      case 'queued':
        return (
          <span className="badge badge-neutral badge-sm gap-1 font-semibold whitespace-nowrap shrink-0">
            <Clock className="w-3 h-3 shrink-0" />
            <span>Đang chờ</span>
          </span>
        );
      default:
        return (
          <span className="badge badge-ghost badge-sm font-medium text-base-content/75 whitespace-nowrap shrink-0">
            Sẵn sàng
          </span>
        );
    }
  };

  const getProgressColorClass = () => {
    if (status === 'failed') return 'progress-error';
    if (status === 'completed') return 'progress-success';
    if (status === 'cancelled') return 'progress-warning';
    return 'progress-primary';
  };

  const showActiveProgressBar =
    isProcessing ||
    status === 'running' ||
    status === 'queued' ||
    status === 'completed' ||
    status === 'failed' ||
    status === 'cancelled';

  return (
    <div
      className={`bg-base-200 border border-base-content/15 shadow-xs px-3.5 py-2.5 text-base-content rounded-xl transition-smooth ${className}`}
    >
      {/* Top Status Strip & Collapsible Section Toggles */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0 flex-1">
          <Activity className={`w-4 h-4 shrink-0 ${isProcessing ? 'text-primary animate-spin' : 'text-primary'}`} />
          <span className="text-xs font-bold text-base-content shrink-0">Trạng thái:</span>
          {getStatusBadge()}
          {statusText && statusText !== 'Sẵn sàng' && (
            <span
              className="truncate font-mono text-[11px] text-base-content/75 ml-1"
              title={statusText}
            >
              {statusText}
            </span>
          )}
        </div>

        {/* Right Controls: Cancel (if running), Toggle OCR/Whisper Config, Toggle Live Logs */}
        <div className="flex items-center gap-1.5 shrink-0">
          {isProcessing && onCancelJob && (
            <button
              type="button"
              onClick={onCancelJob}
              title="Dừng tác vụ đang chạy"
              className="btn btn-sm btn-error btn-outline rounded-full font-semibold cursor-pointer"
            >
              Hủy tác vụ
            </button>
          )}

          {onToggleConfig && (
            <button
              type="button"
              onClick={onToggleConfig}
              aria-expanded={isConfigOpen}
              title={isConfigOpen ? 'Thu gọn cấu hình OCR / Whisper' : 'Mở cấu hình trích xuất OCR / Whisper'}
              className={`btn btn-sm gap-1.5 rounded-full font-semibold cursor-pointer transition-colors ${
                isConfigOpen
                  ? 'btn-primary'
                  : 'btn-outline border-base-content/25 hover:border-primary hover:bg-base-300 hover:text-base-content text-base-content/85'
              }`}
            >
              <Sliders className="w-3.5 h-3.5" />
              <span>Cấu hình OCR / Whisper</span>
              {isConfigOpen ? (
                <ChevronUp className="w-3.5 h-3.5" />
              ) : (
                <ChevronDown className="w-3.5 h-3.5" />
              )}
            </button>
          )}

          <button
            type="button"
            onClick={() => setIsTerminalExpanded(!isTerminalExpanded)}
            aria-expanded={isTerminalExpanded}
            title={isTerminalExpanded ? 'Thu gọn nhật ký hệ thống' : 'Mở nhật ký thời gian thực (Logs)'}
            className={`btn btn-sm gap-1.5 rounded-full font-semibold cursor-pointer transition-colors ${
              isTerminalExpanded
                ? 'btn-primary'
                : 'btn-outline border-base-content/25 hover:border-primary hover:bg-base-300 hover:text-base-content text-base-content/85'
            }`}
          >
            <Terminal className="w-3.5 h-3.5" />
            <span>Nhật ký</span>
            <span className="badge badge-xs font-mono bg-base-100 text-base-content border-base-content/20 px-1.5">
              {logs.length}
            </span>
            {isTerminalExpanded ? (
              <ChevronUp className="w-3.5 h-3.5" />
            ) : (
              <ChevronDown className="w-3.5 h-3.5" />
            )}
          </button>
        </div>
      </div>

      {/* Prominent Progress Bar when running / completed / failed / cancelled */}
      {showActiveProgressBar && (
        <div className="mt-2 flex items-center gap-2.5">
          <progress
            aria-label="Tiến độ xử lý phụ đề"
            className={`progress ${getProgressColorClass()} flex-1 h-2 rounded-full transition-all duration-300`}
            value={clampedProgress}
            max="100"
          />
          <span className="text-xs font-bold font-mono tabular-nums text-primary min-w-[38px] text-right">
            {clampedProgress}%
          </span>
        </div>
      )}

      {/* Collapsible Live Terminal Log */}
      {isTerminalExpanded && (
        <div className="mt-2.5 border-t border-base-content/15 pt-2.5">
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-[11px] font-semibold text-base-content/75 flex items-center gap-1.5">
              <Terminal className="w-3.5 h-3.5 text-primary" />
              <span>Nhật ký thời gian thực (Live Log Stream)</span>
            </span>

            <div className="flex items-center gap-1.5">
              {logs.length > 0 && (
                <>
                  <button
                    type="button"
                    onClick={handleCopyLogs}
                    className="btn btn-ghost btn-sm gap-1 text-xs text-base-content/75 hover:text-base-content cursor-pointer rounded-full"
                    title="Sao chép toàn bộ logs"
                  >
                    {copied ? (
                      <Check className="w-3.5 h-3.5 text-success" />
                    ) : (
                      <Copy className="w-3.5 h-3.5" />
                    )}
                    <span>{copied ? 'Đã sao chép' : 'Sao chép'}</span>
                  </button>

                  {onClearLogs && (
                    <button
                      type="button"
                      onClick={onClearLogs}
                      className="btn btn-ghost btn-sm gap-1 text-xs text-error hover:bg-error/10 cursor-pointer rounded-full"
                      title="Xóa nhật ký hiển thị"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                      <span>Xóa</span>
                    </button>
                  )}
                </>
              )}
            </div>
          </div>

          {/* Terminal Body */}
          <div className="bg-neutral text-neutral-content rounded-lg p-2.5 font-mono text-[11px] leading-relaxed max-h-36 overflow-y-auto border border-base-content/20 shadow-inner">
            {logs.length === 0 ? (
              <p className="text-neutral-content/60 italic">Chưa có thông điệp log nào.</p>
            ) : (
              logs.map((log, index) => {
                let badgeClass = 'text-info';
                if (log.type === 'error') badgeClass = 'text-error font-bold';
                else if (log.type === 'warning') badgeClass = 'text-warning';
                else if (log.type === 'success') badgeClass = 'text-success font-bold';

                return (
                  <div key={log.id || index} className="flex items-start gap-2 py-0.5">
                    <span className="text-neutral-content/60 select-none flex-shrink-0">
                      [{log.timestamp}]
                    </span>
                    <span className={`select-none flex-shrink-0 ${badgeClass}`}>
                      [{(log.type || 'info').toUpperCase()}]
                    </span>
                    <span className="text-neutral-content break-all">{log.message}</span>
                  </div>
                );
              })
            )}
            <div ref={terminalEndRef} />
          </div>
        </div>
      )}
    </div>
  );
};

export default ProgressBar;
