import React, { useState } from 'react';
import {
  History,
  X,
  Clock,
  Eye,
  Download,
  Trash2,
  Mic,
  FileText,
  CheckCircle2,
  Calendar,
} from 'lucide-react';
import type { ExtractionHistoryItem } from '../types/subtitle';
import { getHistoryDownloadUrl } from '../services/api';
import ConfirmModal from './ConfirmModal';

export interface HistoryDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  historyList: ExtractionHistoryItem[];
  onSelectProject: (projectId: string) => void;
  onDeleteHistory: (historyId: string) => void;
  onDeleteHistoryBulk?: (historyIds: string[]) => void | Promise<void>;
  activeProjectId?: string | null;
}

function formatDate(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleString('vi-VN', {
      hour: '2-digit',
      minute: '2-digit',
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
    });
  } catch {
    return iso;
  }
}

function formatDuration(sec: number): string {
  if (!sec || isNaN(sec) || sec <= 0) return '0s';
  const m = Math.floor(sec / 60);
  const s = Math.round(sec % 60);
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

export const HistoryDrawer: React.FC<HistoryDrawerProps> = ({
  isOpen,
  onClose,
  historyList,
  onSelectProject,
  onDeleteHistory,
  onDeleteHistoryBulk,
  activeProjectId,
}) => {
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [isBulkConfirmOpen, setIsBulkConfirmOpen] = useState<boolean>(false);

  if (!isOpen) return null;

  // Chỉ giữ các id còn tồn tại trong danh sách hiện tại
  const validSelected = selectedIds.filter((id) => historyList.some((h) => h.id === id));
  const allSelected = historyList.length > 0 && validSelected.length === historyList.length;
  const someSelected = validSelected.length > 0 && !allSelected;

  const toggleOne = (id: string) =>
    setSelectedIds((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));
  const toggleAll = () => setSelectedIds(allSelected ? [] : historyList.map((h) => h.id));

  return (
    <div className="fixed inset-0 z-50 overflow-hidden bg-blue_slate-100/70 backdrop-blur-xs transition-opacity animate-in fade-in">
      {/* Slide-over panel */}
      <div className="fixed inset-y-0 right-0 flex max-w-full pl-10">
        <div className="w-screen max-w-md bg-base-200 border-l border-blue_slate-500/25 shadow-2xl flex flex-col font-sans">
          {/* Drawer Header */}
          <div className="flex items-center justify-between px-5 py-4 bg-base-300/80 border-b border-blue_slate-500/20">
            <div className="flex items-center gap-3">
              <div className="p-2.5 bg-glaucous-500/15 text-glaucous-600 dark:text-pale_sky-400 rounded-xl border border-blue_slate-500/30 shadow-xs">
                <History className="w-5 h-5" />
              </div>
              <div>
                <h3 className="font-bold text-base text-base-content leading-tight tracking-tight">
                  Lịch sử trích xuất
                </h3>
                <p className="text-xs text-blue_slate-600 dark:text-blue_slate-800 mt-0.5">
                  {historyList.length} lần trích xuất đã lưu
                </p>
              </div>
            </div>
            <button
              type="button"
              onClick={onClose}
              className="btn btn-sm btn-circle btn-ghost text-base-content/70 hover:text-base-content"
              title="Đóng (Close)"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Bulk selection toolbar */}
          {historyList.length > 0 && onDeleteHistoryBulk && (
            <div className="flex items-center justify-between px-5 py-2.5 bg-base-200 border-b border-blue_slate-500/20">
              <label className="label cursor-pointer gap-2 p-0">
                <input
                  type="checkbox"
                  className="checkbox checkbox-primary checkbox-sm"
                  checked={allSelected}
                  ref={(el) => {
                    if (el) el.indeterminate = someSelected;
                  }}
                  onChange={toggleAll}
                />
                <span className="label-text text-xs font-semibold">
                  {validSelected.length > 0 ? `Đã chọn ${validSelected.length}/${historyList.length}` : 'Chọn tất cả'}
                </span>
              </label>
              <button
                type="button"
                disabled={validSelected.length === 0}
                onClick={() => setIsBulkConfirmOpen(true)}
                className="btn btn-xs btn-error gap-1 rounded-lg"
              >
                <Trash2 className="w-3 h-3" />
                Xóa đã chọn
              </button>
            </div>
          )}

          {/* Drawer Content */}
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {historyList.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-64 text-center p-6 text-base-content/50">
                <div className="w-16 h-16 rounded-2xl bg-blue_slate-200/50 border border-blue_slate-500/30 flex items-center justify-center mb-3 shadow-md">
                  <History className="w-8 h-8 text-glaucous-600 dark:text-pale_sky-400" />
                </div>
                <p className="font-semibold text-sm text-base-content">Chưa có lịch sử trích xuất</p>
                <p className="text-xs text-base-content/50 mt-1 max-w-xs leading-relaxed">
                  Mỗi khi hoàn thành trích xuất STT hoặc OCR, video và phụ đề sẽ tự động lưu bền vững tại đây.
                </p>
              </div>
            ) : (
              historyList.map((item) => {
                const isActive = activeProjectId === item.project_id;
                return (
                  <div
                    key={item.id}
                    className={`card rounded-xl transition-smooth ${
                      isActive
                        ? 'bg-glaucous-500/10 border border-glaucous-500 ring-1 ring-glaucous-400/50 shadow-sm'
                        : 'bg-base-100 border border-blue_slate-500/25 hover:border-glaucous-500/40 shadow-xs'
                    }`}
                  >
                    <div className="card-body p-4 gap-2.5">
                      {/* Top row: Checkbox + Video name + Mode badge */}
                      <div className="flex items-start justify-between gap-2">
                        {onDeleteHistoryBulk && (
                          <input
                            type="checkbox"
                            className="checkbox checkbox-primary checkbox-xs mt-0.5 shrink-0"
                            checked={validSelected.includes(item.id)}
                            onChange={() => toggleOne(item.id)}
                            aria-label={`Chọn ${item.project_name}`}
                          />
                        )}
                        <h4
                          className="font-semibold text-xs text-base-content line-clamp-1 tracking-tight flex-1"
                          title={item.project_name}
                        >
                          {item.project_name}
                        </h4>
                        <div className="flex items-center gap-1 shrink-0">
                          {item.mode === 'stt' ? (
                            <span className="badge badge-info badge-sm gap-1 font-medium text-[10px]">
                              <Mic className="w-2.5 h-2.5" /> STT
                            </span>
                          ) : (
                            <span className="badge badge-success badge-sm gap-1 font-medium text-[10px]">
                              <FileText className="w-2.5 h-2.5" /> OCR
                            </span>
                          )}
                          {item.detected_language && (
                            <span className="badge badge-neutral badge-sm text-[10px] uppercase font-mono bg-glaucous-500 text-ghost_white-DEFAULT border-none">
                              {item.detected_language}
                            </span>
                          )}
                        </div>
                      </div>

                      {/* Info row: Date, duration, segment count */}
                      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-base-content/65">
                        <span className="flex items-center gap-1 font-mono">
                          <Calendar className="w-3 h-3 text-glaucous-600 dark:text-pale_sky-400" />
                          {formatDate(item.created_at)}
                        </span>
                        <span className="flex items-center gap-1">
                          <CheckCircle2 className="w-3 h-3 text-success" />
                          {item.segment_count} câu sub
                        </span>
                        {item.duration_sec > 0 && (
                          <span className="flex items-center gap-1 font-mono">
                            <Clock className="w-3 h-3 text-glaucous-600 dark:text-pale_sky-400" />
                            {formatDuration(item.duration_sec)}
                            {item.elapsed_sec > 0 && ` (${item.elapsed_sec}s)`}
                          </span>
                        )}
                      </div>

                      {/* Action buttons */}
                      <div className="flex items-center justify-between pt-2.5 border-t border-blue_slate-500/20 mt-1">
                        <div className="flex items-center gap-1.5">
                          <button
                            type="button"
                            onClick={() => {
                              onSelectProject(item.project_id);
                              onClose();
                            }}
                            className={`btn btn-xs rounded-lg font-semibold cursor-pointer ${
                              isActive ? 'btn-primary' : 'btn-outline border-blue_slate-500/30 text-base-content hover:bg-base-300'
                            }`}
                          >
                            <Eye className="w-3 h-3" />
                            <span>{isActive ? 'Đang mở' : 'Xem lại'}</span>
                          </button>

                          {item.export_srt_path && (
                            <a
                              href={getHistoryDownloadUrl(item.id)}
                              download
                              className="btn btn-xs btn-outline btn-success rounded-lg font-semibold"
                              title="Tải phụ đề .SRT"
                            >
                              <Download className="w-3 h-3" />
                              <span>Tải sub</span>
                            </a>
                          )}
                        </div>

                        <button
                          type="button"
                          onClick={() => setPendingDeleteId(item.id)}
                          className="btn btn-xs btn-ghost text-error hover:bg-error/10 rounded-lg cursor-pointer"
                          title="Xóa khỏi lịch sử"
                        >
                          <Trash2 className="w-3 h-3" />
                        </button>
                      </div>
                    </div>
                  </div>
                );
              })
            )}
          </div>

          {/* Footer */}
          <div className="p-3.5 bg-base-300/60 border-t border-blue_slate-500/20 text-center text-xs text-blue_slate-600 dark:text-blue_slate-800">
            Dữ liệu lịch sử được lưu trữ bền vững trong cơ sở dữ liệu SQLite
          </div>
        </div>
      </div>

      {/* Confirmation Modal using DaisyUI */}
      <ConfirmModal
        isOpen={!!pendingDeleteId}
        title="Xác nhận xóa bản ghi lịch sử"
        message="Bạn có chắc chắn muốn xóa mục lịch sử này khỏi cơ sở dữ liệu? Dữ liệu này sẽ không thể khôi phục."
        confirmLabel="Xóa lịch sử"
        cancelLabel="Hủy"
        onConfirm={() => {
          if (pendingDeleteId) {
            onDeleteHistory(pendingDeleteId);
            setPendingDeleteId(null);
          }
        }}
        onCancel={() => setPendingDeleteId(null)}
      />

      <ConfirmModal
        isOpen={isBulkConfirmOpen}
        title={`Xóa ${validSelected.length} mục lịch sử`}
        message={`Bạn có chắc chắn muốn xóa ${validSelected.length} mục đã chọn khỏi cơ sở dữ liệu? Dữ liệu này sẽ không thể khôi phục.`}
        confirmLabel={`Xóa ${validSelected.length} mục`}
        cancelLabel="Hủy"
        onConfirm={async () => {
          const ids = validSelected;
          setIsBulkConfirmOpen(false);
          setSelectedIds([]);
          await onDeleteHistoryBulk?.(ids);
        }}
        onCancel={() => setIsBulkConfirmOpen(false)}
      />
    </div>
  );
};

export default HistoryDrawer;
