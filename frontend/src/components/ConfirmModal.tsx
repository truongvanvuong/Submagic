import React from 'react';
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
  if (!isOpen) return null;

  return (
    <div className="modal modal-open bg-black/60 backdrop-blur-xs z-[9998] transition-opacity">
      <div className="modal-box max-w-sm p-5 bg-base-200 border border-blue_slate-500/30 rounded-2xl shadow-2xl animate-in zoom-in-95 duration-150">
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
            <p className="text-xs text-base-content/75 mt-1.5 leading-relaxed">{message}</p>
          </div>
        </div>

        <div className="modal-action mt-5 pt-3.5 border-t border-blue_slate-500/20 gap-2">
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
      <div className="modal-backdrop bg-transparent" onClick={onCancel} />
    </div>
  );
};

export default ConfirmModal;
