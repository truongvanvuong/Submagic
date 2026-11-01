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
  className?: string;
}

export const ProgressBar: React.FC<ProgressBarProps> = ({
  isProcessing,
  progress,
  status,
  statusText,
  logs,
  onClearLogs,
  className = '',
}) => {
  const [isTerminalExpanded, setIsTerminalExpanded] = useState<boolean>(true);
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
          <span className="badge badge-info text-info-content badge-sm gap-1 font-medium animate-pulse">
            <span className="w-1.5 h-1.5 rounded-full bg-accent animate-ping" />
            Đang chạy ({clampedProgress}%)
          </span>
        );
      case 'completed':
        return (
          <span className="badge badge-success text-success-content badge-sm gap-1 font-medium">
            <CheckCircle2 className="w-3 h-3" />
            Hoàn tất 100%
          </span>
        );
      case 'failed':
        return (
          <span className="badge badge-error text-error-content badge-sm gap-1 font-medium">
            <XCircle className="w-3 h-3" />
            Thất bại
          </span>
        );
      case 'cancelled':
        return (
          <span className="badge badge-warning text-warning-content badge-sm gap-1 font-medium">
            <AlertTriangle className="w-3 h-3" />
            Đã hủy
          </span>
        );
      case 'queued':
        return (
          <span className="badge badge-neutral badge-sm gap-1 font-medium">
            <Clock className="w-3 h-3" />
            Đang chờ
          </span>
        );
      default:
        return (
          <span className="badge badge-ghost badge-sm font-medium">
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

  return (
    <div
      className={`card bg-base-200 border border-blue_slate-500/25 shadow-lg p-4 text-base-content rounded-xl transition-smooth ${className}`}
    >
      {/* Top Header & Percentage */}
      <div className="flex items-center justify-between gap-2 mb-2">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-primary" />
          <span className="text-xs font-bold text-base-content">Tiến độ thực thi</span>
          {getStatusBadge()}
        </div>

        <span className="text-sm font-bold font-mono text-pale_sky-400">
          {clampedProgress}%
        </span>
      </div>

      {/* daisyUI Progress Bar */}
      <progress
        className={`progress ${getProgressColorClass()} w-full h-2 rounded-full transition-all duration-300`}
        value={clampedProgress}
        max="100"
      />

      {/* Progress Message / Real-time ETA */}
      <div className="flex items-center justify-between mt-2 text-xs text-base-content/70">
        <p className="truncate max-w-[85%] font-mono text-[11px] text-base-content/80">
          {statusText || (isProcessing ? 'Đang xử lý dữ liệu...' : 'Hệ thống đang ở trạng thái nghỉ.')}
        </p>
        <span className="text-[11px] text-base-content/60 font-mono">
          {logs.length} dòng logs
        </span>
      </div>

      {/* Collapsible Live Terminal Log */}
      <div className="mt-3 border-t border-blue_slate-500/20 pt-3">
        <div className="flex items-center justify-between mb-2">
          <button
            type="button"
            onClick={() => setIsTerminalExpanded(!isTerminalExpanded)}
            className="flex items-center gap-1.5 text-xs font-semibold text-base-content/80 hover:text-base-content cursor-pointer transition-colors"
          >
            <Terminal className="w-3.5 h-3.5 text-glaucous-600 dark:text-pale_sky-400" />
            <span>Nhật ký thời gian thực (Live Log Stream)</span>
            {isTerminalExpanded ? (
              <ChevronUp className="w-3.5 h-3.5 text-base-content/50" />
            ) : (
              <ChevronDown className="w-3.5 h-3.5 text-base-content/50" />
            )}
          </button>

          <div className="flex items-center gap-1.5">
            {logs.length > 0 && (
              <>
                <button
                  type="button"
                  onClick={handleCopyLogs}
                  className="btn btn-ghost btn-xs gap-1 text-[11px] text-base-content/70 hover:text-base-content cursor-pointer rounded-md"
                  title="Sao chép toàn bộ logs"
                >
                  {copied ? (
                    <Check className="w-3 h-3 text-success" />
                  ) : (
                    <Copy className="w-3 h-3" />
                  )}
                  <span>{copied ? 'Đã sao chép' : 'Sao chép'}</span>
                </button>

                {onClearLogs && (
                  <button
                    type="button"
                    onClick={onClearLogs}
                    className="btn btn-ghost btn-xs gap-1 text-[11px] text-error hover:bg-error/10 cursor-pointer rounded-md"
                    title="Xóa nhật ký hiển thị"
                  >
                    <Trash2 className="w-3 h-3" />
                    <span>Xóa</span>
                  </button>
                )}
              </>
            )}
          </div>
        </div>

        {/* Terminal Body */}
        {isTerminalExpanded && (
          <div className="bg-neutral text-neutral-content rounded-lg p-3 font-mono text-[11px] leading-relaxed max-h-48 overflow-y-auto border border-blue_slate-500/30 shadow-inner">
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
        )}
      </div>
    </div>
  );
};

export default ProgressBar;
