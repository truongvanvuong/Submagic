import React from 'react';
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

export interface HistoryDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  historyList: ExtractionHistoryItem[];
  onSelectProject: (projectId: string) => void;
  onDeleteHistory: (historyId: string) => void;
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
  activeProjectId,
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-hidden bg-ash-grey-950/60 backdrop-blur-xs transition-opacity animate-in fade-in">
      {/* Slide-over panel */}
      <div className="fixed inset-y-0 right-0 flex max-w-full pl-10">
        <div className="w-screen max-w-md bg-base-100 border-l border-base-200 shadow-2xl flex flex-col">
          {/* Drawer Header */}
          <div className="flex items-center justify-between px-5 py-4 bg-base-200/50 border-b border-base-300">
            <div className="flex items-center gap-2.5">
              <div className="p-2 bg-primary/10 text-primary rounded-lg">
                <History className="w-5 h-5" />
              </div>
              <div>
                <h3 className="font-bold text-base text-base-content leading-tight">
                  Lịch sử trích xuất
                </h3>
                <p className="text-xs text-base-content/60">
                  {historyList.length} lần trích xuất đã lưu
                </p>
              </div>
            </div>
            <button
              type="button"
              onClick={onClose}
              className="btn btn-sm btn-circle btn-ghost"
              title="Đóng (Close)"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Drawer Content */}
          <div className="flex-1 overflow-y-auto p-4 space-y-3">
            {historyList.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-64 text-center p-6 text-base-content/50">
                <div className="w-16 h-16 rounded-full bg-base-200 flex items-center justify-center mb-3">
                  <History className="w-8 h-8 text-base-content/40" />
                </div>
                <p className="font-medium text-sm text-base-content/80">Chưa có lịch sử trích xuất</p>
                <p className="text-xs text-base-content/50 mt-1 max-w-xs">
                  Mỗi khi hoàn thành trích xuất STT hoặc OCR, video và phụ đề sẽ tự động lưu bền vững tại đây.
                </p>
              </div>
            ) : (
              historyList.map((item) => {
                const isActive = activeProjectId === item.project_id;
                return (
                  <div
                    key={item.id}
                    className={`card bg-base-100 border transition-all ${
                      isActive
                        ? 'border-primary ring-1 ring-primary/40 shadow-sm'
                        : 'border-base-200 hover:border-base-300 shadow-xs'
                    }`}
                  >
                    <div className="card-body p-4 gap-2">
                      {/* Top row: Video name + Mode badge */}
                      <div className="flex items-start justify-between gap-2">
                        <h4
                          className="font-semibold text-xs text-base-content line-clamp-1"
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
                            <span className="badge badge-neutral badge-sm text-[10px] uppercase font-mono">
                              {item.detected_language}
                            </span>
                          )}
                        </div>
                      </div>

                      {/* Info row: Date, duration, segment count */}
                      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-base-content/60">
                        <span className="flex items-center gap-1">
                          <Calendar className="w-3 h-3 text-base-content/40" />
                          {formatDate(item.created_at)}
                        </span>
                        <span className="flex items-center gap-1">
                          <CheckCircle2 className="w-3 h-3 text-success" />
                          {item.segment_count} câu sub
                        </span>
                        {item.duration_sec > 0 && (
                          <span className="flex items-center gap-1">
                            <Clock className="w-3 h-3 text-base-content/40" />
                            {formatDuration(item.duration_sec)}
                            {item.elapsed_sec > 0 && ` (${item.elapsed_sec}s)`}
                          </span>
                        )}
                      </div>

                      {/* Action buttons */}
                      <div className="flex items-center justify-between pt-2 border-t border-base-200 mt-1">
                        <div className="flex items-center gap-1.5">
                          <button
                            type="button"
                            onClick={() => {
                              onSelectProject(item.project_id);
                              onClose();
                            }}
                            className={`btn btn-xs ${
                              isActive ? 'btn-primary' : 'btn-outline btn-primary'
                            }`}
                          >
                            <Eye className="w-3 h-3" />
                            <span>{isActive ? 'Đang mở' : 'Xem lại'}</span>
                          </button>

                          {item.export_srt_path && (
                            <a
                              href={getHistoryDownloadUrl(item.id)}
                              download
                              className="btn btn-xs btn-outline btn-success"
                              title="Tải phụ đề .SRT"
                            >
                              <Download className="w-3 h-3" />
                              <span>Tải sub</span>
                            </a>
                          )}
                        </div>

                        <button
                          type="button"
                          onClick={() => {
                            if (window.confirm(`Bạn có chắc muốn xóa lịch sử này?`)) {
                              onDeleteHistory(item.id);
                            }
                          }}
                          className="btn btn-xs btn-ghost text-error hover:bg-error/10"
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
          <div className="p-3 bg-base-200/50 border-t border-base-300 text-center text-xs text-base-content/50">
            Dữ liệu lịch sử được lưu trữ bền vững trong cơ sở dữ liệu SQLite
          </div>
        </div>
      </div>
    </div>
  );
};

export default HistoryDrawer;
