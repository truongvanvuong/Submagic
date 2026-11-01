import React, { useState, useEffect } from 'react';
import {
  Mic,
  FileText,
  Settings2,
  Play,
  Square,
  Loader2,
  Cpu,
  Key,
  Eye,
  EyeOff,
  Download,
  Sliders,
} from 'lucide-react';
import type {
  ExtractionMode,
  HardwareInfo,
  ModelCatalogResponse,
  OCRJobRequest,
  ProjectDetail,
  STTJobRequest,
  LLMJobRequest,
} from '../types/subtitle';
import { AUDIO_LANGUAGES, OCR_LANGUAGES } from '../constants/languages';

export interface ControlPanelProps {
  project: ProjectDetail | null;
  hardwareInfo?: HardwareInfo | null;
  modelCatalog?: ModelCatalogResponse | null;
  isProcessing: boolean;
  onStartSTT: (options: STTJobRequest) => Promise<void> | void;
  onStartOCR: (options: OCRJobRequest) => Promise<void> | void;
  onStartLLM?: (options: LLMJobRequest) => Promise<void> | void;
  onCancelJob: () => Promise<void> | void;
  onOpenExport?: () => void;
  className?: string;
}

export const ControlPanel: React.FC<ControlPanelProps> = ({
  project,
  hardwareInfo,
  modelCatalog,
  isProcessing,
  onStartSTT,
  onStartOCR,
  onCancelJob,
  onOpenExport,
  className = '',
}) => {
  // Mode selection: 'stt' (Whisper audio) vs 'ocr' (Hardsub video OCR)
  const [activeMode, setActiveMode] = useState<ExtractionMode>('stt');

  // STT Settings - Requirement 4: default language is 'auto'
  const [sttModel, setSttModel] = useState<STTJobRequest['model_size']>('small');
  const [sttLanguage, setSttLanguage] = useState<string>('auto');
  const [sttDevice, setSttDevice] = useState<'auto' | 'cuda' | 'cpu'>('auto');
  const [sttVadFilter, setSttVadFilter] = useState<boolean>(true);
  const [sttWordTimestamps, setSttWordTimestamps] = useState<boolean>(true);
  const [sttMaxChars, setSttMaxChars] = useState<number>(42);
  const [sttMaxLines, setSttMaxLines] = useState<number>(2);

  // OCR Settings - Requirement 4: default language is 'auto', Requirement 6: default 3.0 FPS
  const [ocrEngine, setOcrEngine] = useState<OCRJobRequest['engine']>('auto');
  const [ocrLanguage, setOcrLanguage] = useState<OCRJobRequest['language']>('auto');
  const [ocrSampleFps, setOcrSampleFps] = useState<number>(3.0);
  const [ocrLumaThreshold, setOcrLumaThreshold] = useState<number>(185);
  const [ocrStabilityFrames, setOcrStabilityFrames] = useState<number>(1);
  const [ocrSaveCrops, setOcrSaveCrops] = useState<boolean>(true);

  // Gemini Cloud API Key
  const [geminiApiKey, setGeminiApiKey] = useState<string>('');
  const [showApiKey, setShowApiKey] = useState<boolean>(false);

  // Advanced settings accordion toggle
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);

  // Load saved API key and default preferences from localStorage
  useEffect(() => {
    try {
      const savedKey = localStorage.getItem('gemini_api_key');
      if (savedKey) {
        setGeminiApiKey(savedKey);
      }
    } catch {
      // LocalStorage access error handling
    }
  }, []);

  const handleApiKeyChange = (val: string) => {
    setGeminiApiKey(val);
    try {
      localStorage.setItem('gemini_api_key', val);
    } catch {
      // Ignore storage errors
    }
  };

  // Sync mode with project if project provides a default mode
  useEffect(() => {
    if (project?.mode) {
      if (project.mode === 'ocr') {
        setActiveMode('ocr');
      } else {
        setActiveMode('stt');
      }
    }
  }, [project?.id, project?.mode]);

  // Handle start action
  const handleExecute = () => {
    if (!project) return;

    if (activeMode === 'stt') {
      const req: STTJobRequest = {
        project_id: project.id,
        model_size: sttModel,
        language: sttLanguage === 'auto' ? null : sttLanguage,
        device: sttDevice,
        vad_filter: sttVadFilter,
        word_timestamps: sttWordTimestamps,
        max_chars_per_line: sttMaxChars,
        max_lines_per_segment: sttMaxLines,
        gemini_api_key: geminiApiKey || undefined,
      };
      onStartSTT(req);
    } else {
      const req: OCRJobRequest = {
        project_id: project.id,
        language: ocrLanguage,
        engine: ocrEngine,
        sample_fps: ocrSampleFps,
        roi_config: project.roi_config,
        luma_threshold: ocrLumaThreshold,
        stability_frames: ocrStabilityFrames,
        save_crop_images: ocrSaveCrops,
        gemini_api_key: geminiApiKey || undefined,
      };
      onStartOCR(req);
    }
  };

  const hasCuda = hardwareInfo?.cuda_available || false;
  const gpuName = hardwareInfo?.gpu_names?.[0] || (hasCuda ? 'NVIDIA GPU' : 'CPU Mode');

  return (
    <div className={`card bg-base-100 border border-base-200 shadow-xs p-4 sm:p-5 text-base-content ${className}`}>
      {/* Header & Mode Switcher */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pb-3 border-b border-base-200">
        <div>
          <h2 className="text-sm sm:text-base font-bold flex items-center gap-2 text-base-content">
            <Sliders className="w-4 h-4 text-primary" />
            Cấu hình Trích xuất Phụ đề
          </h2>
          <p className="text-xs text-base-content/60 mt-0.5">
            Lựa chọn thuật toán AI và tham số xử lý tối ưu cho video
          </p>
        </div>

        {/* Hardware Status Badge */}
        <div className="badge badge-outline gap-1.5 py-3 px-3 text-xs font-medium">
          <Cpu className={`w-3.5 h-3.5 ${hasCuda ? 'text-success' : 'text-base-content/50'}`} />
          <span className="truncate max-w-[160px]">{gpuName}</span>
          {hasCuda && <span className="badge badge-success badge-xs">CUDA</span>}
        </div>
      </div>

      {/* Mode Tabs using daisyUI tabs-boxed */}
      <div className="tabs tabs-boxed bg-base-200/80 p-1 mt-4 grid grid-cols-2">
        <button
          type="button"
          onClick={() => setActiveMode('stt')}
          disabled={isProcessing}
          className={`tab tab-sm font-semibold transition-all gap-2 h-9 ${
            activeMode === 'stt' ? 'tab-active !bg-primary !text-primary-content shadow-xs' : 'text-base-content/70'
          }`}
        >
          <Mic className="w-4 h-4" />
          Âm thanh (STT Whisper)
        </button>

        <button
          type="button"
          onClick={() => setActiveMode('ocr')}
          disabled={isProcessing}
          className={`tab tab-sm font-semibold transition-all gap-2 h-9 ${
            activeMode === 'ocr' ? 'tab-active !bg-primary !text-primary-content shadow-xs' : 'text-base-content/70'
          }`}
        >
          <FileText className="w-4 h-4" />
          Phụ đề cứng (Hardsub OCR)
        </button>
      </div>

      {/* Main Settings Form */}
      <div className="mt-4 space-y-4">
        {activeMode === 'stt' ? (
          /* ================= STT Mode Controls ================= */
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {/* Model Selection */}
            <div className="form-control w-full">
              <label className="label py-1">
                <span className="label-text text-xs font-medium">Mô hình Faster-Whisper</span>
              </label>
              <select
                value={sttModel}
                onChange={(e) => setSttModel(e.target.value as STTJobRequest['model_size'])}
                disabled={isProcessing}
                className="select select-bordered select-sm w-full text-xs"
              >
                <option value="tiny">Whisper Tiny (~75 MB - Nhanh nhất)</option>
                <option value="base">Whisper Base (~145 MB)</option>
                <option value="small">Whisper Small (~465 MB - Khuyên dùng)</option>
                <option value="medium">Whisper Medium (~1.5 GB)</option>
                <option value="large-v3-turbo">Whisper Large-v3 Turbo (Tối ưu GPU)</option>
                <option value="large-v3">Whisper Large-v3 (~3.1 GB - Tốt nhất)</option>
              </select>
            </div>

            {/* Language Selector (Auto default) */}
            <div className="form-control w-full">
              <label className="label py-1">
                <span className="label-text text-xs font-medium">Ngôn ngữ Giọng nói</span>
              </label>
              <select
                value={sttLanguage}
                onChange={(e) => setSttLanguage(e.target.value)}
                disabled={isProcessing}
                className="select select-bordered select-sm w-full text-xs"
              >
                {AUDIO_LANGUAGES.map((lang) => (
                  <option key={lang.code} value={lang.code}>
                    {lang.flag} {lang.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Device Selector */}
            <div className="form-control w-full">
              <label className="label py-1">
                <span className="label-text text-xs font-medium">Thiết bị Tính toán</span>
              </label>
              <select
                value={sttDevice}
                onChange={(e) => setSttDevice(e.target.value as 'auto' | 'cuda' | 'cpu')}
                disabled={isProcessing}
                className="select select-bordered select-sm w-full text-xs"
              >
                <option value="auto">Tự động (Ưu tiên GPU CUDA)</option>
                <option value="cuda" disabled={!hasCuda}>
                  NVIDIA GPU CUDA {hasCuda ? '✓' : '(Không khả dụng)'}
                </option>
                <option value="cpu">CPU INT8 (Ổn định mọi máy)</option>
              </select>
            </div>
          </div>
        ) : (
          /* ================= OCR Mode Controls ================= */
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {/* OCR Engine Selection */}
            <div className="form-control w-full">
              <label className="label py-1">
                <span className="label-text text-xs font-medium">Công cụ Nhận diện (Engine)</span>
              </label>
              <select
                value={ocrEngine}
                onChange={(e) => setOcrEngine(e.target.value as OCRJobRequest['engine'])}
                disabled={isProcessing}
                className="select select-bordered select-sm w-full text-xs"
              >
                <option value="auto">Tự động (Khuyên dùng)</option>
                <option value="rapidocr">RapidOCR (ONNX PP-OCRv4 - Rất nhanh)</option>
                <option value="vietocr">RapidOCR + VietOCR (Chuẩn dấu Tiếng Việt)</option>
                <option value="gemini">Gemini 2.5 Flash Vision (Cloud AI)</option>
              </select>
            </div>

            {/* OCR Language Selector (Auto default) */}
            <div className="form-control w-full">
              <label className="label py-1">
                <span className="label-text text-xs font-medium">Ngôn ngữ Chữ viết</span>
              </label>
              <select
                value={ocrLanguage}
                onChange={(e) => setOcrLanguage(e.target.value as OCRJobRequest['language'])}
                disabled={isProcessing}
                className="select select-bordered select-sm w-full text-xs"
              >
                {OCR_LANGUAGES.map((lang) => (
                  <option key={lang.code} value={lang.code}>
                    {lang.flag} {lang.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Sample FPS (Optimized default 3.0 FPS) */}
            <div className="form-control w-full">
              <label className="label py-1 flex justify-between items-center">
                <span className="label-text text-xs font-medium">Tần suất quét FPS</span>
                <span className="badge badge-primary badge-sm font-mono">{ocrSampleFps} FPS</span>
              </label>
              <div className="px-1 pt-1.5">
                <input
                  type="range"
                  min="1.0"
                  max="10.0"
                  step="0.5"
                  value={ocrSampleFps}
                  onChange={(e) => setOcrSampleFps(parseFloat(e.target.value))}
                  disabled={isProcessing}
                  className="range range-primary range-xs w-full"
                />
              </div>
            </div>
          </div>
        )}

        {/* Collapsible Advanced Settings Accordion */}
        <div className="pt-2 border-t border-base-200">
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="btn btn-ghost btn-xs text-xs font-medium gap-1 text-base-content/70 hover:text-base-content"
          >
            <Settings2 className="w-3.5 h-3.5" />
            <span>{showAdvanced ? 'Ẩn tùy chọn nâng cao' : 'Hiện tùy chọn nâng cao & Gemini API'}</span>
          </button>

          {showAdvanced && (
            <div className="mt-3 p-4 bg-base-200/50 rounded-xl border border-base-200 space-y-3">
              {/* STT Advanced Parameters */}
              {activeMode === 'stt' && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <label className="label cursor-pointer justify-start gap-3 py-1">
                    <input
                      type="checkbox"
                      checked={sttVadFilter}
                      onChange={(e) => setSttVadFilter(e.target.checked)}
                      disabled={isProcessing}
                      className="checkbox checkbox-primary checkbox-sm"
                    />
                    <span className="label-text text-xs select-none">
                      Lọc khoảng lặng Silero VAD (Giảm ảo giác)
                    </span>
                  </label>

                  <label className="label cursor-pointer justify-start gap-3 py-1">
                    <input
                      type="checkbox"
                      checked={sttWordTimestamps}
                      onChange={(e) => setSttWordTimestamps(e.target.checked)}
                      disabled={isProcessing}
                      className="checkbox checkbox-primary checkbox-sm"
                    />
                    <span className="label-text text-xs select-none">
                      Căn chỉnh mốc thời gian từng từ (Word-level)
                    </span>
                  </label>

                  <div className="form-control">
                    <label className="label py-1">
                      <span className="label-text text-xs">
                        Ký tự tối đa mỗi dòng: <span className="font-mono font-bold">{sttMaxChars}</span>
                      </span>
                    </label>
                    <input
                      type="range"
                      min="24"
                      max="72"
                      value={sttMaxChars}
                      onChange={(e) => setSttMaxChars(parseInt(e.target.value, 10))}
                      disabled={isProcessing}
                      className="range range-primary range-xs"
                    />
                  </div>

                  <div className="form-control">
                    <label className="label py-1">
                      <span className="label-text text-xs">
                        Số dòng tối đa mỗi câu: <span className="font-mono font-bold">{sttMaxLines}</span>
                      </span>
                    </label>
                    <input
                      type="range"
                      min="1"
                      max="3"
                      value={sttMaxLines}
                      onChange={(e) => setSttMaxLines(parseInt(e.target.value, 10))}
                      disabled={isProcessing}
                      className="range range-primary range-xs"
                    />
                  </div>
                </div>
              )}

              {/* OCR Advanced Parameters */}
              {activeMode === 'ocr' && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div className="form-control">
                    <label className="label py-1">
                      <span className="label-text text-xs">
                        Ngưỡng lọc sáng Luma: <span className="font-mono font-bold">{ocrLumaThreshold}</span>
                      </span>
                    </label>
                    <input
                      type="range"
                      min="80"
                      max="240"
                      value={ocrLumaThreshold}
                      onChange={(e) => setOcrLumaThreshold(parseInt(e.target.value, 10))}
                      disabled={isProcessing}
                      className="range range-primary range-xs"
                    />
                  </div>

                  <div className="form-control">
                    <label className="label py-1">
                      <span className="label-text text-xs">
                        Khung hình ổn định: <span className="font-mono font-bold">{ocrStabilityFrames} frame</span>
                      </span>
                    </label>
                    <input
                      type="range"
                      min="0"
                      max="4"
                      value={ocrStabilityFrames}
                      onChange={(e) => setOcrStabilityFrames(parseInt(e.target.value, 10))}
                      disabled={isProcessing}
                      className="range range-primary range-xs"
                    />
                  </div>

                  <label className="label cursor-pointer justify-start gap-3 py-1 sm:col-span-2">
                    <input
                      type="checkbox"
                      checked={ocrSaveCrops}
                      onChange={(e) => setOcrSaveCrops(e.target.checked)}
                      disabled={isProcessing}
                      className="checkbox checkbox-primary checkbox-sm"
                    />
                    <span className="label-text text-xs select-none">
                      Lưu và hiển thị ảnh crop chữ gốc để đối chiếu khi biên tập
                    </span>
                  </label>
                </div>
              )}

              {/* Gemini Cloud API Key Input */}
              <div className="pt-3 border-t border-base-200">
                <label className="label py-1">
                  <span className="label-text text-xs font-medium flex items-center gap-1.5">
                    <Key className="w-3.5 h-3.5 text-warning" />
                    Google Gemini API Key (Tùy chọn)
                  </span>
                  <span className="label-text-alt text-[11px] text-base-content/50">
                    Dùng cho AI OCR / Sửa lỗi / Dịch thuật
                  </span>
                </label>
                <div className="relative">
                  <input
                    type={showApiKey ? 'text' : 'password'}
                    placeholder="AIzaSy..."
                    value={geminiApiKey}
                    onChange={(e) => handleApiKeyChange(e.target.value)}
                    disabled={isProcessing}
                    className="input input-bordered input-sm w-full font-mono text-xs pr-10"
                  />
                  <button
                    type="button"
                    onClick={() => setShowApiKey(!showApiKey)}
                    className="absolute right-2.5 top-1/2 -translate-y-1/2 text-base-content/40 hover:text-base-content"
                  >
                    {showApiKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Action Buttons Bar */}
        <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-base-200">
          <div className="flex items-center gap-2">
            {!isProcessing ? (
              <button
                type="button"
                onClick={handleExecute}
                disabled={!project}
                className="btn btn-primary btn-sm gap-2"
              >
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>Bắt đầu trích xuất ({activeMode.toUpperCase()})</span>
              </button>
            ) : (
              <div className="flex items-center gap-2">
                <button type="button" disabled className="btn btn-primary btn-sm gap-2 loading">
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>Đang xử lý {activeMode.toUpperCase()}...</span>
                </button>

                <button
                  type="button"
                  onClick={onCancelJob}
                  className="btn btn-error btn-outline btn-sm gap-1.5"
                  title="Dừng tiến trình hiện tại"
                >
                  <Square className="w-3.5 h-3.5 fill-current" />
                  <span>Hủy tác vụ</span>
                </button>
              </div>
            )}
          </div>

          {onOpenExport && (
            <button
              type="button"
              onClick={onOpenExport}
              disabled={!project || (project.segments && project.segments.length === 0)}
              className="btn btn-outline btn-sm gap-1.5"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Xuất file phụ đề</span>
              {project?.segments && project.segments.length > 0 && (
                <span className="badge badge-sm badge-neutral font-mono">{project.segments.length}</span>
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default ControlPanel;
