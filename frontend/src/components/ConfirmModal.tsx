import React, { useEffect } from 'react';
import { createPortal } from 'react-dom';
import { AlertTriangle, Trash2 } from 'lucide-react';

export interface ConfirmModalProps {
  isOpen: boolean;
  title?: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  isDanger?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

export const ConfirmModal: React.FC<ConfirmModalProps> = ({
  isOpen,
  title = 'Xác nhận xóa',
  message,
  confirmLabel = 'Xóa vĩnh viễn',
  cancelLabel = 'Hủy bỏ',
  isDanger = true,
  onConfirm,
  onCancel,
}) => {
  // Close on Escape key
  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onCancel();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onCancel]);

  if (!isOpen || typeof document === 'undefined') return null;

  const content = (
    <div className="fixed inset-0 z-[99999] flex items-center justify-center p-4 bg-black/60 backdrop-blur-xs transition-opacity animate-in fade-in duration-150">
      <div className="w-full max-w-sm p-5 bg-base-200 border border-blue_slate-500/30 rounded-2xl shadow-2xl animate-in zoom-in-95 duration-150 relative z-10 font-sans">
        <div className="flex items-start gap-3.5">
          <div
            className={`w-10 h-10 rounded-xl flex items-center justify-center shrink-0 ${
              isDanger
                ? 'bg-error/15 text-error border border-error/30'
                : 'bg-warning/15 text-warning border border-warning/30'
            }`}
          >
            {isDanger ? <Trash2 className="w-5 h-5" /> : <AlertTriangle className="w-5 h-5" />}
          </div>

          <div className="flex-1 min-w-0">
            <h3 className="font-bold text-sm text-base-content tracking-tight">{title}</h3>
            <p className="text-xs text-base-content/75 mt-1.5 leading-relaxed whitespace-pre-line">{message}</p>
          </div>
        </div>

        <div className="flex items-center justify-end mt-5 pt-3.5 border-t border-blue_slate-500/20 gap-2">
          <button
            type="button"
            onClick={onCancel}
            className="btn btn-sm btn-ghost rounded-lg font-medium text-xs cursor-pointer"
          >
            {cancelLabel}
          </button>
          <button
            type="button"
            onClick={onConfirm}
            className={`btn btn-sm rounded-lg font-semibold text-xs shadow-sm cursor-pointer ${
              isDanger ? 'btn-error text-white' : 'btn-primary'
            }`}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
      <div className="fixed inset-0 bg-transparent" onClick={onCancel} />
    </div>
  );

  return createPortal(content, document.body);
};

export default ConfirmModal;
