import React, { useState, useEffect } from 'react';
import { Loader2, Eye, EyeOff } from 'lucide-react';
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
  isProcessing,
  onStartSTT,
  onStartOCR,
  onCancelJob,
  onOpenExport,
  className = '',
}) => {
  // Mode selection: 'stt' (Whisper audio) vs 'ocr' (Hardsub video OCR)
  const [activeMode, setActiveMode] = useState<ExtractionMode>('stt');

  // STT Settings
  const [sttModel, setSttModel] = useState<STTJobRequest['model_size']>('small');
  const [sttLanguage, setSttLanguage] = useState<string>('auto');
  const [sttDevice, setSttDevice] = useState<'auto' | 'cuda' | 'cpu'>('auto');
  const [sttVadFilter, setSttVadFilter] = useState<boolean>(true);
  const [sttWordTimestamps, setSttWordTimestamps] = useState<boolean>(true);
  const [sttMaxChars, setSttMaxChars] = useState<number>(42);
  const [sttMaxLines, setSttMaxLines] = useState<number>(2);

  // OCR Settings
  const [ocrEngine, setOcrEngine] = useState<OCRJobRequest['engine']>('auto');
  const [ocrLanguage, setOcrLanguage] = useState<OCRJobRequest['language']>('auto');
  const [ocrSampleFps, setOcrSampleFps] = useState<number>(6.0);
  const [ocrLumaThreshold, setOcrLumaThreshold] = useState<number>(185);
  const [ocrStabilityFrames, setOcrStabilityFrames] = useState<number>(1);
  const [ocrSaveCrops, setOcrSaveCrops] = useState<boolean>(true);

  // Gemini Cloud API Key
  const [geminiApiKey, setGeminiApiKey] = useState<string>('');
  const [showApiKey, setShowApiKey] = useState<boolean>(false);

  // Advanced settings accordion toggle
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);

  // Load saved API key from localStorage
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
    <div
      className={`rounded-xl bg-base-200 border border-blue_slate-500/25 p-4 sm:p-5 text-base-content shadow-lg transition-smooth ${className}`}
    >
      {/* 1. Header & Hardware Status */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pb-3.5 border-b border-blue_slate-500/20">
        <div>
          <h2 className="text-xs sm:text-sm font-bold uppercase tracking-wider text-base-content">
            Cấu hình trích xuất phụ đề
          </h2>
          <p className="text-[11px] text-blue_slate-600 dark:text-blue_slate-800 mt-0.5">
            Lựa chọn thuật toán AI và tham số xử lý tối ưu cho video
          </p>
        </div>

        {/* Minimal Hardware Pill */}
        <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-lg bg-base-300 border border-blue_slate-500/30 text-[11px] text-base-content/85 font-mono">
          <span
            className={`w-2 h-2 rounded-full ${
              hasCuda ? 'bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.5)]' : 'bg-blue_slate-400'
            }`}
          />
          <span className="truncate max-w-[170px]">{gpuName}</span>
          {hasCuda && (
            <span className="text-[10px] uppercase font-mono px-1 py-0.5 rounded bg-glaucous-500 text-ghost_white-DEFAULT border border-glaucous-600">
              CUDA
            </span>
          )}
        </div>
      </div>

      {/* 2. Mode Segmented Switcher */}
      <div className="mt-3.5 p-1 rounded-xl bg-base-300/80 border border-blue_slate-500/30 grid grid-cols-2 gap-1.5">
        <button
          type="button"
          onClick={() => setActiveMode('stt')}
          disabled={isProcessing}
          className={`py-2 px-3 rounded-lg text-xs font-semibold tracking-wide transition-smooth text-center cursor-pointer ${
            activeMode === 'stt'
              ? 'bg-primary text-primary-content shadow-sm'
              : 'text-base-content/70 hover:text-base-content hover:bg-base-200/60'
          }`}
        >
          Âm thanh (Whisper STT)
        </button>

        <button
          type="button"
          onClick={() => setActiveMode('ocr')}
          disabled={isProcessing}
          className={`py-2 px-3 rounded-lg text-xs font-semibold tracking-wide transition-smooth text-center cursor-pointer ${
            activeMode === 'ocr'
              ? 'bg-primary text-primary-content shadow-sm'
              : 'text-base-content/70 hover:text-base-content hover:bg-base-200/60'
          }`}
        >
          Phụ đề cứng (Video OCR)
        </button>
      </div>

      {/* 3. Core Parameter Controls */}
      <div className="mt-4">
        {activeMode === 'stt' ? (
          /* ================= STT Core Controls ================= */
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {/* Model Selection */}
            <div>
              <label className="block text-[11px] font-medium text-base-content/85 mb-1.5">
                Mô hình Whisper
              </label>
              <select
                value={sttModel}
                onChange={(e) => setSttModel(e.target.value as STTJobRequest['model_size'])}
                disabled={isProcessing}
                className="w-full h-8 px-2.5 rounded-lg bg-base-100 border border-blue_slate-500/30 text-base-content text-xs focus:outline-none focus:border-glaucous-500 transition-colors cursor-pointer"
              >
                <option value="tiny">Tiny (~75 MB - Rất nhanh)</option>
                <option value="base">Base (~145 MB)</option>
                <option value="small">Small (~465 MB - Khuyên dùng)</option>
                <option value="medium">Medium (~1.5 GB)</option>
                <option value="large-v3-turbo">Large-v3 Turbo (Tối ưu GPU)</option>
                <option value="large-v3">Large-v3 (~3.1 GB - Tốt nhất)</option>
              </select>
            </div>

            {/* Language Selector */}
            <div>
              <label className="block text-[11px] font-medium text-base-content/85 mb-1.5">
                Ngôn ngữ giọng nói
              </label>
              <select
                value={sttLanguage}
                onChange={(e) => setSttLanguage(e.target.value)}
                disabled={isProcessing}
                className="w-full h-8 px-2.5 rounded-lg bg-base-100 border border-blue_slate-500/30 text-base-content text-xs focus:outline-none focus:border-glaucous-500 transition-colors cursor-pointer"
              >
                {AUDIO_LANGUAGES.map((lang) => (
                  <option key={lang.code} value={lang.code}>
                    {lang.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Device Selector */}
            <div>
              <label className="block text-[11px] font-medium text-base-content/85 mb-1.5">
                Thiết bị xử lý
              </label>
              <select
                value={sttDevice}
                onChange={(e) => setSttDevice(e.target.value as 'auto' | 'cuda' | 'cpu')}
                disabled={isProcessing}
                className="w-full h-8 px-2.5 rounded-lg bg-base-100 border border-blue_slate-500/30 text-base-content text-xs focus:outline-none focus:border-glaucous-500 transition-colors cursor-pointer"
              >
                <option value="auto">Tự động (Ưu tiên CUDA)</option>
                <option value="cuda" disabled={!hasCuda}>
                  GPU CUDA {hasCuda ? '✓' : '(Không khả dụng)'}
                </option>
                <option value="cpu">CPU INT8 (Mọi máy tính)</option>
              </select>
            </div>
          </div>
        ) : (
          /* ================= OCR Core Controls ================= */
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {/* OCR Engine Selection */}
            <div>
              <label className="block text-[11px] font-medium text-base-content/85 mb-1.5">
                Công cụ OCR
              </label>
              <select
                value={ocrEngine}
                onChange={(e) => setOcrEngine(e.target.value as OCRJobRequest['engine'])}
                disabled={isProcessing}
                className="w-full h-8 px-2.5 rounded-lg bg-base-100 border border-blue_slate-500/30 text-base-content text-xs focus:outline-none focus:border-glaucous-500 transition-colors cursor-pointer"
              >
                <option value="auto">Tự động (Khuyên dùng)</option>
                <option value="rapidocr">RapidOCR (ONNX - Nhanh)</option>
                <option value="vietocr">RapidOCR + VietOCR (Chuẩn dấu Việt)</option>
                <option value="gemini">Gemini Flash Vision (Cloud AI)</option>
              </select>
            </div>

            {/* OCR Language Selector */}
            <div>
              <label className="block text-[11px] font-medium text-base-content/85 mb-1.5">
                Ngôn ngữ chữ viết
              </label>
              <select
                value={ocrLanguage}
                onChange={(e) => setOcrLanguage(e.target.value as OCRJobRequest['language'])}
                disabled={isProcessing}
                className="w-full h-8 px-2.5 rounded-lg bg-base-100 border border-blue_slate-500/30 text-base-content text-xs focus:outline-none focus:border-glaucous-500 transition-colors cursor-pointer"
              >
                {OCR_LANGUAGES.map((lang) => (
                  <option key={lang.code} value={lang.code}>
                    {lang.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Sample FPS Slider */}
            <div>
              <div className="flex justify-between items-center mb-1.5">
                <label className="text-[11px] font-medium text-base-content/85">
                  Tần suất quét FPS
                </label>
                <span className="font-mono text-[11px] px-1.5 py-0.5 rounded bg-base-300 border border-blue_slate-500/30 text-glaucous-600 dark:text-pale_sky-400 font-semibold">
                  {ocrSampleFps.toFixed(1)} FPS
                </span>
              </div>
              <div className="pt-1.5">
                <input
                  type="range"
                  min="1.0"
                  max="10.0"
                  step="0.5"
                  value={ocrSampleFps}
                  onChange={(e) => setOcrSampleFps(parseFloat(e.target.value))}
                  disabled={isProcessing}
                  className="range range-primary range-xs w-full cursor-pointer disabled:opacity-40"
                />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* 4. Collapsible Advanced Settings & Cloud Key Drawer */}
      <div className="mt-4 pt-3 border-t border-blue_slate-500/20">
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="text-[11px] font-semibold uppercase tracking-wider text-glaucous-600 dark:text-pale_sky-400 hover:text-base-content transition-colors flex items-center gap-1.5 cursor-pointer"
        >
          <span
            className={`text-[9px] transition-transform duration-200 ${
              showAdvanced ? 'rotate-90' : 'rotate-0'
            }`}
          >
            ▶
          </span>
          <span>{showAdvanced ? 'Ẩn tùy chọn nâng cao & Gemini API' : 'Tùy chọn nâng cao & Gemini API'}</span>
        </button>

        {showAdvanced && (
          <div className="mt-3 p-3.5 rounded-xl bg-base-300/60 border border-blue_slate-500/25 space-y-3.5">
            {/* STT Advanced Parameters */}
            {activeMode === 'stt' && (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
                <label className="flex items-center gap-2.5 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={sttVadFilter}
                    onChange={(e) => setSttVadFilter(e.target.checked)}
                    disabled={isProcessing}
                    className="checkbox checkbox-primary checkbox-xs rounded"
                  />
                  <span className="text-xs text-base-content/85 font-medium">
                    Lọc khoảng lặng Silero VAD (Giảm ảo giác)
                  </span>
                </label>

                <label className="flex items-center gap-2.5 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={sttWordTimestamps}
                    onChange={(e) => setSttWordTimestamps(e.target.checked)}
                    disabled={isProcessing}
                    className="checkbox checkbox-primary checkbox-xs rounded"
                  />
                  <span className="text-xs text-base-content/85 font-medium">
                    Mốc thời gian từng từ (Word-level timestamps)
                  </span>
                </label>

                <div>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-xs text-base-content/85">Ký tự tối đa mỗi dòng</span>
                    <span className="font-mono text-xs font-semibold text-pale_sky-400">{sttMaxChars}</span>
                  </div>
                  <input
                    type="range"
                    min="24"
                    max="72"
                    value={sttMaxChars}
                    onChange={(e) => setSttMaxChars(parseInt(e.target.value, 10))}
                    disabled={isProcessing}
                    className="range range-primary range-xs w-full cursor-pointer disabled:opacity-40"
                  />
                </div>

                <div>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-xs text-base-content/85">Số dòng tối đa mỗi câu</span>
                    <span className="font-mono text-xs font-semibold text-pale_sky-400">{sttMaxLines}</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="3"
                    value={sttMaxLines}
                    onChange={(e) => setSttMaxLines(parseInt(e.target.value, 10))}
                    disabled={isProcessing}
                    className="range range-primary range-xs w-full cursor-pointer disabled:opacity-40"
                  />
                </div>
              </div>
            )}

            {/* OCR Advanced Parameters */}
            {activeMode === 'ocr' && (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-xs text-base-content/85">Ngưỡng lọc sáng Luma</span>
                    <span className="font-mono text-xs font-semibold text-pale_sky-400">{ocrLumaThreshold}</span>
                  </div>
                  <input
                    type="range"
                    min="80"
                    max="240"
                    value={ocrLumaThreshold}
                    onChange={(e) => setOcrLumaThreshold(parseInt(e.target.value, 10))}
                    disabled={isProcessing}
                    className="range range-primary range-xs w-full cursor-pointer disabled:opacity-40"
                  />
                </div>

                <div>
                  <div className="flex justify-between items-center mb-1">
                    <span className="text-xs text-base-content/85">Khung hình ổn định</span>
                    <span className="font-mono text-xs font-semibold text-pale_sky-400">{ocrStabilityFrames} frames</span>
                  </div>
                  <input
                    type="range"
                    min="0"
                    max="4"
                    value={ocrStabilityFrames}
                    onChange={(e) => setOcrStabilityFrames(parseInt(e.target.value, 10))}
                    disabled={isProcessing}
                    className="range range-primary range-xs w-full cursor-pointer disabled:opacity-40"
                  />
                </div>

                <label className="flex items-center gap-2.5 cursor-pointer select-none sm:col-span-2 pt-1">
                  <input
                    type="checkbox"
                    checked={ocrSaveCrops}
                    onChange={(e) => setOcrSaveCrops(e.target.checked)}
                    disabled={isProcessing}
                    className="checkbox checkbox-primary checkbox-xs rounded"
                  />
                  <span className="text-xs text-base-content/85 font-medium">
                    Lưu ảnh crop phụ đề gốc để đối chiếu khi biên tập
                  </span>
                </label>
              </div>
            )}

            {/* Gemini Cloud API Key */}
            <div className="pt-3 border-t border-blue_slate-500/20">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1 mb-1.5">
                <label className="text-xs font-medium text-base-content/90">
                  Google Gemini API Key
                </label>
                <span className="text-[11px] text-blue_slate-600 dark:text-blue_slate-800">
                  Dùng cho OCR Vision / AI Sửa lỗi / Dịch thuật
                </span>
              </div>
              <div className="relative">
                <input
                  type={showApiKey ? 'text' : 'password'}
                  placeholder="AIzaSy..."
                  value={geminiApiKey}
                  onChange={(e) => handleApiKeyChange(e.target.value)}
                  disabled={isProcessing}
                  className="w-full h-8 px-2.5 pr-9 rounded-lg bg-base-100 border border-blue_slate-500/30 font-mono text-xs text-base-content placeholder:text-blue_slate-600/50 dark:placeholder:text-blue_slate-800/50 focus:outline-none focus:border-glaucous-500 transition-colors"
                />
                <button
                  type="button"
                  onClick={() => setShowApiKey(!showApiKey)}
                  className="absolute right-2.5 top-1/2 -translate-y-1/2 text-blue_slate-600 dark:text-blue_slate-800 hover:text-base-content transition-colors cursor-pointer"
                  title={showApiKey ? 'Ẩn khóa' : 'Hiện khóa'}
                >
                  {showApiKey ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* 5. Action Toolbar */}
      <div className="mt-4 pt-3.5 border-t border-blue_slate-500/20 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2.5">
          {!isProcessing ? (
            <button
              type="button"
              onClick={handleExecute}
              disabled={!project}
              className="btn btn-sm btn-primary rounded-lg font-semibold shadow-md cursor-pointer disabled:cursor-not-allowed"
            >
              Bắt đầu trích xuất ({activeMode.toUpperCase()})
            </button>
          ) : (
            <div className="flex items-center gap-2">
              <button
                type="button"
                disabled
                className="btn btn-sm btn-primary rounded-lg font-medium flex items-center gap-2 cursor-wait"
              >
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>Đang xử lý {activeMode.toUpperCase()}...</span>
              </button>

              <button
                type="button"
                onClick={onCancelJob}
                className="btn btn-sm btn-error btn-outline rounded-lg font-medium cursor-pointer"
                title="Dừng tác vụ đang chạy"
              >
                Hủy tác vụ
              </button>
            </div>
          )}
        </div>

        {onOpenExport && (
          <button
            type="button"
            onClick={onOpenExport}
            disabled={!project || (project.segments && project.segments.length === 0)}
            className="btn btn-sm btn-outline border-blue_slate-500/30 hover:border-glaucous-500 hover:bg-base-300 text-base-content/85 rounded-lg flex items-center gap-2 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed font-medium"
          >
            <span>Xuất phụ đề</span>
            {project?.segments && project.segments.length > 0 && (
              <span className="badge badge-xs bg-glaucous-500 text-ghost_white-DEFAULT border-none px-1.5 font-mono">
                {project.segments.length}
              </span>
            )}
          </button>
        )}
      </div>
    </div>
  );
};

export default ControlPanel;
