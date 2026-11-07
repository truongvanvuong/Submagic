import React, { useState, useEffect } from 'react';
import {
  CheckCircle2,
  AlertTriangle,
  Info,
  XCircle,
  X,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';

export interface ToastItem {
  id: string;
  type: 'success' | 'info' | 'warning' | 'error';
  title: string;
  message?: string;
  details?: string[];
  duration?: number;
}

export interface ToastContainerProps {
  toasts: ToastItem[];
  onDismiss: (id: string) => void;
}

export const ToastContainer: React.FC<ToastContainerProps> = ({ toasts, onDismiss }) => {
  if (toasts.length === 0) return null;

  return (
    <div
      role="region"
      aria-label="Thông báo hệ thống"
      className="fixed top-4 right-4 z-[99999] flex flex-col gap-2.5 max-w-[calc(100vw-2rem)] sm:max-w-md w-full pointer-events-none font-sans"
    >
      {toasts.map((toast) => (
        <ToastCard key={toast.id} toast={toast} onDismiss={() => onDismiss(toast.id)} />
      ))}
    </div>
  );
};

interface ToastCardProps {
  toast: ToastItem;
  onDismiss: () => void;
}

const ToastCard: React.FC<ToastCardProps> = ({ toast, onDismiss }) => {
  const [showDetails, setShowDetails] = useState<boolean>(false);

  useEffect(() => {
    // If user opens details, don't auto-dismiss quickly
    const dismissDuration = showDetails ? 15000 : toast.duration || 4500;
    const timer = setTimeout(() => {
      onDismiss();
    }, dismissDuration);

    return () => clearTimeout(timer);
  }, [toast.id, toast.duration, showDetails, onDismiss]);

  const getAlertConfig = () => {
    switch (toast.type) {
      case 'success':
        return {
          bgClass: 'bg-emerald-600 text-white border-emerald-500/60 shadow-emerald-950/20',
          icon: <CheckCircle2 className="w-5 h-5 shrink-0 text-white" />,
        };
      case 'warning':
        return {
          bgClass: 'bg-amber-600 text-white border-amber-500/60 shadow-amber-950/20',
          icon: <AlertTriangle className="w-5 h-5 shrink-0 text-white" />,
        };
      case 'error':
        return {
          bgClass: 'bg-red-600 text-white border-red-500/60 shadow-red-950/20',
          icon: <XCircle className="w-5 h-5 shrink-0 text-white" />,
        };
      case 'info':
      default:
        return {
          bgClass: 'bg-blue-600 text-white border-blue-500/60 shadow-blue-950/20',
          icon: <Info className="w-5 h-5 shrink-0 text-white" />,
        };
    }
  };

  const { bgClass, icon } = getAlertConfig();
  const hasDetails = Boolean(toast.details && toast.details.length > 0);

  return (
    <div
      role="alert"
      className={`${bgClass} shadow-2xl rounded-2xl p-3.5 border flex flex-col pointer-events-auto transition-all duration-200 animate-in fade-in slide-in-from-top-2 max-w-full`}
    >
      <div className="flex items-start gap-3 w-full">
        <div className="mt-0.5">{icon}</div>

        <div className="flex-1 min-w-0 pr-1">
          <h4 className="font-bold text-xs tracking-tight leading-snug text-white">
            {toast.title}
          </h4>
          {toast.message && (
            <p className="text-[12px] font-medium text-white/95 mt-0.5 leading-relaxed break-words whitespace-normal">
              {toast.message}
            </p>
          )}

          {hasDetails && (
            <button
              type="button"
              onClick={() => setShowDetails((prev) => !prev)}
              className="mt-1.5 inline-flex items-center gap-1 text-[11px] font-semibold text-white/90 underline hover:text-white cursor-pointer select-none"
            >
              <span>{showDetails ? 'Ẩn chi tiết' : `Xem chi tiết (${toast.details!.length})`}</span>
              {showDetails ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            </button>
          )}
        </div>

        <button
          type="button"
          onClick={onDismiss}
          className="w-6 h-6 flex items-center justify-center -mr-1 -mt-1 text-white/70 hover:text-white hover:bg-white/15 rounded-lg transition-colors cursor-pointer shrink-0"
          title="Đóng thông báo"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Collapsible Details Box */}
      {hasDetails && showDetails && (
        <div className="mt-2.5 pt-2 border-t border-white/20 w-full animate-in fade-in duration-150">
          <div className="max-h-36 overflow-y-auto bg-black/25 text-white/95 p-2 rounded-xl text-[11px] font-mono space-y-1 divide-y divide-white/10 shadow-inner">
            {toast.details!.map((item, idx) => (
              <div key={idx} className="pt-1 first:pt-0 break-words whitespace-normal leading-tight">
                {item}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default ToastContainer;
