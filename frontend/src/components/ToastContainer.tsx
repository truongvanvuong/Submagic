import React, { useEffect } from 'react';
import { CheckCircle2, AlertTriangle, Info, XCircle, X } from 'lucide-react';

export interface ToastItem {
  id: string;
  type: 'success' | 'info' | 'warning' | 'error';
  title: string;
  message?: string;
  duration?: number;
}

export interface ToastContainerProps {
  toasts: ToastItem[];
  onDismiss: (id: string) => void;
}

export const ToastContainer: React.FC<ToastContainerProps> = ({ toasts, onDismiss }) => {
  if (toasts.length === 0) return null;

  return (
    <div className="toast toast-top toast-end z-[9999] fixed top-4 right-4 p-0 flex flex-col gap-2.5 max-w-sm w-full pointer-events-none animate-in fade-in slide-in-from-top-2 duration-200">
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
  useEffect(() => {
    const timer = setTimeout(() => {
      onDismiss();
    }, toast.duration || 4000);

    return () => clearTimeout(timer);
  }, [toast.id, toast.duration, onDismiss]);

  const getAlertConfig = () => {
    switch (toast.type) {
      case 'success':
        return {
          alertClass: 'alert alert-success text-success-content border-success/40',
          icon: <CheckCircle2 className="w-5 h-5 shrink-0" />,
        };
      case 'warning':
        return {
          alertClass: 'alert alert-warning text-warning-content border-warning/40',
          icon: <AlertTriangle className="w-5 h-5 shrink-0" />,
        };
      case 'error':
        return {
          alertClass: 'alert alert-error text-error-content border-error/40',
          icon: <XCircle className="w-5 h-5 shrink-0" />,
        };
      case 'info':
      default:
        return {
          alertClass: 'alert alert-info text-info-content border-info/40',
          icon: <Info className="w-5 h-5 shrink-0" />,
        };
    }
  };

  const { alertClass, icon } = getAlertConfig();

  return (
    <div
      role="alert"
      className={`${alertClass} shadow-2xl rounded-xl p-3.5 border flex items-start gap-3 pointer-events-auto transition-all duration-200 hover:shadow-xl`}
    >
      <div className="mt-0.5">{icon}</div>

      <div className="flex-1 min-w-0 pr-1">
        <h4 className="font-bold text-xs tracking-tight leading-snug">{toast.title}</h4>
        {toast.message && (
          <p className="text-[11px] opacity-90 mt-0.5 leading-relaxed break-words">{toast.message}</p>
        )}
      </div>

      <button
        type="button"
        onClick={onDismiss}
        className="btn btn-ghost btn-xs btn-square -mr-1 -mt-1 opacity-70 hover:opacity-100 hover:bg-black/10 rounded-md"
        title="Đóng thông báo"
      >
        <X className="w-3.5 h-3.5" />
      </button>
    </div>
  );
};

export default ToastContainer;
