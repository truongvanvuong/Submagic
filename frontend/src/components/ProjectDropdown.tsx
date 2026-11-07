import React, { useState, useRef, useEffect, useMemo } from 'react';
import {
  FolderOpen,
  ChevronDown,
  Trash2,
  Search,
  Check,
  X,
  FileText,
  Mic,
  ArrowUpDown,
  AlertTriangle,
} from 'lucide-react';
import type { ProjectDetail, ProjectSummary } from '../types/subtitle';
import ConfirmModal from './ConfirmModal';

export interface ProjectDropdownProps {
  projects: ProjectSummary[];
  activeProject: ProjectDetail | null;
  onSelectProject: (projectId: string | null) => void;
  onDeleteProject: (projectId: string) => void | Promise<void>;
  onBulkDeleteProjects?: (projectIds: string[]) => void | Promise<void>;
  isProcessing?: boolean;
  disabled?: boolean;
}

type SortOption = 'newest' | 'oldest' | 'name';

export const ProjectDropdown: React.FC<ProjectDropdownProps> = ({
  projects,
  activeProject,
  onSelectProject,
  onDeleteProject,
  onBulkDeleteProjects,
  isProcessing = false,
  disabled = false,
}) => {
  const [isOpen, setIsOpen] = useState<boolean>(false);
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [sortOrder, setSortOrder] = useState<SortOption>('newest');
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [displayLimit, setDisplayLimit] = useState<number>(40);

  // Modals state
  const [pendingDeleteProject, setPendingDeleteProject] = useState<ProjectSummary | null>(null);
  const [isBulkDeleteModalOpen, setIsBulkDeleteModalOpen] = useState<boolean>(false);

  // Ref for Shift+Click range selection
  const lastSelectedIndexRef = useRef<number | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const listRef = useRef<HTMLDivElement | null>(null);

  // Close dropdown when clicking outside or pressing Escape
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (pendingDeleteProject || isBulkDeleteModalOpen) return;
        setIsOpen(false);
      }
    };

    if (isOpen) {
      document.addEventListener('mousedown', handleClickOutside);
      window.addEventListener('keydown', handleKeyDown);
    }
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen, pendingDeleteProject, isBulkDeleteModalOpen]);

  // Reset displayLimit when query or sort changes
  useEffect(() => {
    setDisplayLimit(40);
  }, [searchQuery, sortOrder]);

  // Clean selectedIds to only include valid project IDs
  useEffect(() => {
    setSelectedIds((prev) => prev.filter((id) => projects.some((p) => p.id === id)));
  }, [projects]);

  // Sort and filter projects
  const sortedAndFilteredProjects = useMemo(() => {
    let result = [...projects];

    // Filter by search query
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase().trim();
      result = result.filter(
        (p) =>
          p.name.toLowerCase().includes(q) ||
          p.mode.toLowerCase().includes(q)
      );
    }

    // Sort projects
    if (sortOrder === 'newest') {
      result.sort((a, b) => b.id.localeCompare(a.id));
    } else if (sortOrder === 'oldest') {
      result.sort((a, b) => a.id.localeCompare(b.id));
    } else if (sortOrder === 'name') {
      result.sort((a, b) => a.name.localeCompare(b.name, 'vi', { numeric: true }));
    }

    return result;
  }, [projects, searchQuery, sortOrder]);

  // Displayed slice (Infinite Scroll / Lazy Windowing for 300+ items)
  const displayedProjects = useMemo(() => {
    return sortedAndFilteredProjects.slice(0, displayLimit);
  }, [sortedAndFilteredProjects, displayLimit]);

  // Infinite scroll trigger
  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    const target = e.currentTarget;
    if (target.scrollTop + target.clientHeight >= target.scrollHeight - 60) {
      if (displayLimit < sortedAndFilteredProjects.length) {
        setDisplayLimit((prev) => Math.min(prev + 40, sortedAndFilteredProjects.length));
      }
    }
  };

  // Check if a project cannot be selected/deleted (active project or currently running)
  const isProjectProtected = (p: ProjectSummary): boolean => {
    return p.status === 'running' || (activeProject !== null && activeProject.id === p.id);
  };

  // Only selectable projects for Select All calculation
  const selectableVisibleIds = useMemo(() => {
    return sortedAndFilteredProjects
      .filter((p) => !isProjectProtected(p))
      .map((p) => p.id);
  }, [sortedAndFilteredProjects, activeProject]);

  const allSelectableSelected =
    selectableVisibleIds.length > 0 &&
    selectableVisibleIds.every((id) => selectedIds.includes(id));

  const someSelectableSelected =
    selectableVisibleIds.some((id) => selectedIds.includes(id)) && !allSelectableSelected;

  // Toggle select all visible selectable projects
  const handleToggleSelectAll = () => {
    if (allSelectableSelected) {
      setSelectedIds((prev) => prev.filter((id) => !selectableVisibleIds.includes(id)));
    } else {
      setSelectedIds((prev) => Array.from(new Set([...prev, ...selectableVisibleIds])));
    }
  };

  // Single select with Shift+Click range support
  const handleToggleSelectOne = (id: string, index: number, shiftKey: boolean) => {
    const targetProject = sortedAndFilteredProjects[index];
    if (targetProject && isProjectProtected(targetProject)) {
      return;
    }

    if (shiftKey && lastSelectedIndexRef.current !== null) {
      const start = Math.min(lastSelectedIndexRef.current, index);
      const end = Math.max(lastSelectedIndexRef.current, index);
      const rangeSelectableIds = sortedAndFilteredProjects
        .slice(start, end + 1)
        .filter((p) => !isProjectProtected(p))
        .map((p) => p.id);
      setSelectedIds((prev) => Array.from(new Set([...prev, ...rangeSelectableIds])));
    } else {
      setSelectedIds((prev) =>
        prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
      );
      lastSelectedIndexRef.current = index;
    }
  };

  const handleClearSelection = () => {
    setSelectedIds([]);
    lastSelectedIndexRef.current = null;
  };

  // Delete handlers
  const handleDeleteSingleClick = (e: React.MouseEvent, project: ProjectSummary) => {
    e.stopPropagation();
    if (isProjectProtected(project)) return;
    setPendingDeleteProject(project);
  };

  const handleConfirmDeleteSingle = async () => {
    if (pendingDeleteProject) {
      const pid = pendingDeleteProject.id;
      setPendingDeleteProject(null);
      await onDeleteProject(pid);
    }
  };

  const handleConfirmBulkDelete = async () => {
    const idsToDelete = [...selectedIds];
    setIsBulkDeleteModalOpen(false);
    setSelectedIds([]);
    if (onBulkDeleteProjects) {
      await onBulkDeleteProjects(idsToDelete);
    } else {
      for (const id of idsToDelete) {
        await onDeleteProject(id);
      }
    }
  };

  // Bulk delete confirmation summary
  const bulkDeleteSummary = useMemo(() => {
    if (selectedIds.length === 0) return '';
    const selectedProjects = projects.filter((p) => selectedIds.includes(p.id));
    const firstNames = selectedProjects.slice(0, 4).map((p) => `• ${p.name}`);
    if (selectedProjects.length > 4) {
      firstNames.push(`• ... và ${selectedProjects.length - 4} dự án khác`);
    }
    return firstNames.join('\n');
  }, [projects, selectedIds]);

  return (
    <div ref={containerRef} className="relative inline-block text-left">
      {/* Trigger Button on Header */}
      <div className="flex items-center gap-1 bg-base-300/60 hover:bg-base-300 border border-blue_slate-500/25 rounded-lg px-2.5 py-1 transition-smooth max-w-full">
        <button
          type="button"
          disabled={disabled}
          onClick={() => setIsOpen((prev) => !prev)}
          className="flex items-center gap-1.5 text-xs font-medium text-base-content focus:outline-none max-w-[180px] xs:max-w-[220px] sm:max-w-[280px] md:max-w-[340px] cursor-pointer select-none"
          title={activeProject ? `${activeProject.name} (${activeProject.mode.toUpperCase()})` : 'Chọn dự án'}
        >
          <FolderOpen className="w-3.5 h-3.5 text-base-content/60 shrink-0" />
          <span className="truncate">
            {activeProject ? activeProject.name : '-- Chưa chọn video --'}
          </span>
          {activeProject && (
            <span className="badge badge-xs text-[9px] font-mono uppercase bg-glaucous-500 text-ghost_white-DEFAULT border-none shrink-0">
              {activeProject.mode}
            </span>
          )}
          <ChevronDown
            className={`w-3.5 h-3.5 text-base-content/50 shrink-0 transition-transform duration-150 ${
              isOpen ? 'rotate-180' : ''
            }`}
          />
        </button>
      </div>

      {/* Dropdown Menu Popover */}
      {isOpen && (
        <div
          className="absolute left-0 mt-2 w-96 max-w-[calc(100vw-1.5rem)] bg-base-200 border border-blue_slate-500/30 rounded-2xl shadow-2xl ring-1 ring-black/10 dark:ring-white/10 z-[60] overflow-hidden flex flex-col font-sans animate-in fade-in zoom-in-95 duration-100"
          style={{ minWidth: '320px' }}
        >
          {/* Top Bar: Search input & Sort button */}
          <div className="p-3 border-b border-blue_slate-500/20 bg-base-300/80 space-y-2.5">
            <div className="flex items-center gap-2">
              <div className="relative flex-1">
                <Search className="w-3.5 h-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 text-base-content/50" />
                <input
                  type="text"
                  autoFocus
                  placeholder="Tìm dự án..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="input input-xs w-full pl-8 pr-7 bg-base-100 border-blue_slate-500/30 focus:border-primary rounded-lg text-xs"
                />
                {searchQuery && (
                  <button
                    type="button"
                    onClick={() => setSearchQuery('')}
                    className="absolute right-2 top-1/2 -translate-y-1/2 text-base-content/50 hover:text-base-content"
                    title="Xóa tìm kiếm"
                  >
                    <X className="w-3 h-3" />
                  </button>
                )}
              </div>

              {/* Sort Order Toggle */}
              <button
                type="button"
                onClick={() =>
                  setSortOrder((prev) =>
                    prev === 'newest' ? 'oldest' : prev === 'oldest' ? 'name' : 'newest'
                  )
                }
                title={`Sắp xếp: ${
                  sortOrder === 'newest'
                    ? 'Mới nhất'
                    : sortOrder === 'oldest'
                    ? 'Cũ nhất'
                    : 'Theo tên A-Z'
                }`}
                className="btn btn-xs btn-ghost border border-blue_slate-500/30 bg-base-100 rounded-lg gap-1 text-[11px] font-medium shrink-0"
              >
                <ArrowUpDown className="w-3 h-3 text-glaucous-600 dark:text-pale_sky-400" />
                <span>
                  {sortOrder === 'newest'
                    ? 'Mới nhất'
                    : sortOrder === 'oldest'
                    ? 'Cũ nhất'
                    : 'Tên A-Z'}
                </span>
              </button>
            </div>

            {/* Bulk Selection Toolbar (Always present like History Drawer) */}
            <div className="flex items-center justify-between pt-1 border-t border-blue_slate-500/20 text-xs">
              <label className="flex items-center gap-2 cursor-pointer select-none">
                <input
                  type="checkbox"
                  checked={allSelectableSelected}
                  ref={(el) => {
                    if (el) el.indeterminate = someSelectableSelected;
                  }}
                  onChange={handleToggleSelectAll}
                  disabled={selectableVisibleIds.length === 0}
                  className="checkbox checkbox-primary checkbox-xs rounded"
                />
                <span className="text-[11px] font-semibold text-base-content/85">
                  {selectedIds.length > 0
                    ? `Đã chọn ${selectedIds.length} dự án`
                    : 'Chọn tất cả'}
                </span>
              </label>

              <div className="flex items-center gap-1.5">
                {selectedIds.length > 0 && (
                  <button
                    type="button"
                    onClick={handleClearSelection}
                    className="text-[11px] text-base-content/60 hover:text-base-content px-1.5 py-0.5 rounded cursor-pointer"
                  >
                    Bỏ chọn
                  </button>
                )}
                <button
                  type="button"
                  disabled={selectedIds.length === 0}
                  onClick={() => setIsBulkDeleteModalOpen(true)}
                  className="btn btn-xs btn-error gap-1 text-[11px] font-bold rounded-lg shadow-xs disabled:opacity-40 cursor-pointer"
                  title="Xóa các dự án đã chọn"
                >
                  <Trash2 className="w-3 h-3" />
                  <span>Xóa đã chọn{selectedIds.length > 0 ? ` (${selectedIds.length})` : ''}</span>
                </button>
              </div>
            </div>
          </div>

          {/* Projects List Container (Virtual scroll / infinite windowing) */}
          <div
            ref={listRef}
            onScroll={handleScroll}
            className="max-h-80 overflow-y-auto p-1.5 space-y-1 divide-y divide-blue_slate-500/10"
          >
            {/* Unselect Option */}
            {!searchQuery && (
              <div
                onClick={() => {
                  onSelectProject(null);
                  setIsOpen(false);
                }}
                className={`flex items-center justify-between px-3 py-2 rounded-xl text-xs cursor-pointer transition-colors ${
                  !activeProject
                    ? 'bg-primary/10 text-primary font-semibold'
                    : 'text-base-content/70 hover:bg-base-300 hover:text-base-content'
                }`}
              >
                <span>-- Không chọn dự án --</span>
                {!activeProject && <Check className="w-3.5 h-3.5 text-primary" />}
              </div>
            )}

            {displayedProjects.length === 0 ? (
              <div className="p-6 text-center text-xs text-base-content/50">
                {searchQuery
                  ? `Không tìm thấy dự án nào khớp với "${searchQuery}"`
                  : 'Chưa có dự án nào'}
              </div>
            ) : (
              displayedProjects.map((p, idx) => {
                const isActive = activeProject?.id === p.id;
                const isRunning = p.status === 'running';
                const isProtected = isActive || isRunning;
                const isSelected = selectedIds.includes(p.id);

                return (
                  <div
                    key={p.id}
                    onClick={() => {
                      onSelectProject(p.id);
                      setIsOpen(false);
                    }}
                    className={`group flex items-center justify-between px-2.5 py-2 rounded-xl text-xs cursor-pointer transition-colors ${
                      isSelected
                        ? 'bg-primary/15 border border-primary/40'
                        : isActive
                        ? 'bg-glaucous-500/15 border border-glaucous-500/40 text-base-content font-medium shadow-xs'
                        : 'hover:bg-base-300/80 text-base-content/85'
                    }`}
                  >
                    {/* Left: Checkbox + Mode Icon + Name info */}
                    <div className="flex items-center gap-2.5 min-w-0 flex-1 pr-2">
                      <input
                        type="checkbox"
                        checked={isSelected}
                        disabled={isProtected}
                        onChange={(e) => {
                          e.stopPropagation();
                          handleToggleSelectOne(
                            p.id,
                            idx,
                            (e.nativeEvent as MouseEvent).shiftKey
                          );
                        }}
                        onClick={(e) => e.stopPropagation()}
                        className="checkbox checkbox-primary checkbox-xs rounded shrink-0 disabled:opacity-30 disabled:cursor-not-allowed"
                        title={
                          isRunning
                            ? 'Dự án đang trong quá trình xử lý, không thể xóa'
                            : isActive
                            ? 'Dự án đang được mở trong trình chỉnh sửa'
                            : 'Chọn dự án (giữ Shift để chọn dải)'
                        }
                        aria-label={`Chọn dự án ${p.name}`}
                      />

                      {p.mode === 'stt' ? (
                        <Mic className="w-3.5 h-3.5 text-glaucous-600 dark:text-pale_sky-400 shrink-0" />
                      ) : (
                        <FileText className="w-3.5 h-3.5 text-success shrink-0" />
                      )}

                      <div className="min-w-0 flex-1">
                        <div
                          className="truncate font-medium text-xs text-base-content"
                          title={p.name}
                        >
                          {p.name}
                        </div>
                        <div className="flex items-center gap-1.5 text-[10px] text-base-content/55 mt-0.5">
                          <span className="uppercase font-mono font-semibold">
                            {p.mode}
                          </span>
                          {p.segment_count > 0 && (
                            <>
                              <span>•</span>
                              <span>{p.segment_count} câu sub</span>
                            </>
                          )}
                          {isActive && (
                            <>
                              <span>•</span>
                              <span className="text-primary font-semibold">Đang mở</span>
                            </>
                          )}
                          {isRunning && (
                            <>
                              <span>•</span>
                              <span className="text-warning font-semibold animate-pulse">
                                Đang xử lý
                              </span>
                            </>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Right: Active Check & Delete Button */}
                    <div className="flex items-center gap-1 shrink-0">
                      {isActive && (
                        <Check className="w-3.5 h-3.5 text-glaucous-600 dark:text-pale_sky-400 mr-1" />
                      )}

                      <button
                        type="button"
                        onClick={(e) => handleDeleteSingleClick(e, p)}
                        disabled={isProtected}
                        className="w-7 h-7 flex items-center justify-center text-error hover:bg-error/20 bg-error/10 border border-error/25 rounded-lg transition-all cursor-pointer disabled:opacity-20 disabled:cursor-not-allowed shadow-2xs"
                        title={
                          isRunning
                            ? 'Dự án đang trong quá trình xử lý, không thể xóa'
                            : isActive
                            ? 'Dự án đang mở trong trình chỉnh sửa, không thể xóa từ đây'
                            : `Xóa dự án "${p.name}"`
                        }
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>
                );
              })
            )}

            {/* Infinite Scroll indicator */}
            {displayLimit < sortedAndFilteredProjects.length && (
              <div className="p-2 text-center text-[11px] text-base-content/50">
                Đang hiển thị {displayLimit}/{sortedAndFilteredProjects.length} dự án • Cuộn xuống để xem thêm
              </div>
            )}
          </div>

          {/* Footer Stats Bar */}
          <div className="px-3 py-2 bg-base-300/60 border-t border-blue_slate-500/20 text-[11px] text-base-content/60 flex justify-between items-center">
            <span>
              Tổng cộng: <strong>{projects.length}</strong> dự án
            </span>
            <span className="text-[10px] text-base-content/50">
              Giữ Shift + click để chọn dải
            </span>
          </div>
        </div>
      )}

      {/* Confirmation Modal for Single Project Deletion (React Portal) */}
      <ConfirmModal
        isOpen={!!pendingDeleteProject}
        title="Xác nhận xóa dự án"
        message={
          pendingDeleteProject
            ? `Bạn có chắc chắn muốn xóa dự án "${pendingDeleteProject.name}"?\n\nToàn bộ video, ảnh trích xuất, phụ đề và dữ liệu liên quan trên đĩa sẽ bị xóa sạch và không thể hoàn tác.`
            : ''
        }
        confirmLabel="Xóa vĩnh viễn"
        cancelLabel="Hủy"
        onConfirm={handleConfirmDeleteSingle}
        onCancel={() => setPendingDeleteProject(null)}
      />

      {/* Confirmation Modal for Bulk Project Deletion (React Portal) */}
      <ConfirmModal
        isOpen={isBulkDeleteModalOpen}
        title={`Xác nhận xóa ${selectedIds.length} dự án đã chọn`}
        message={`Bạn có chắc chắn muốn xóa vĩnh viễn ${selectedIds.length} dự án đã chọn khỏi hệ thống?\n\n${bulkDeleteSummary}\n\nToàn bộ video, ảnh frame trích xuất, phụ đề và dữ liệu trên đĩa của các dự án này sẽ bị xóa sạch và không thể hoàn tác.`}
        confirmLabel={`Xóa vĩnh viễn ${selectedIds.length} dự án`}
        cancelLabel="Hủy"
        onConfirm={handleConfirmBulkDelete}
        onCancel={() => setIsBulkDeleteModalOpen(false)}
      />
    </div>
  );
};

export default ProjectDropdown;
