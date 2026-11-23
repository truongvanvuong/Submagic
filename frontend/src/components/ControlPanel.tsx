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
      className={`rounded-xl bg-base-200 border border-base-content/15 p-3.5 text-base-content shadow-xs transition-smooth ${className}`}
    >
      {/* 1. Top Row: Mode Segmented Switcher + Hardware Pill + Primary Execute Button */}
      <div className="flex flex-wrap items-center justify-between gap-2.5 pb-3 border-b border-base-content/15">
        {/* Mode Segmented Switcher */}
        <div
          role="tablist"
          aria-label="Chế độ trích xuất phụ đề"
          className="p-1 rounded-full bg-base-300/80 border border-base-content/20 inline-grid grid-cols-2 gap-1"
        >
          <button
            type="button"
            role="tab"
            aria-selected={activeMode === 'stt'}
            onClick={() => setActiveMode('stt')}
            disabled={isProcessing}
            className={`btn btn-sm border-none rounded-full text-xs font-semibold tracking-wide transition-smooth cursor-pointer ${
              activeMode === 'stt'
                ? 'btn-primary shadow-xs'
                : 'btn-ghost text-base-content/75 hover:text-base-content'
            }`}
          >
            Âm thanh (Whisper STT)
          </button>

          <button
            type="button"
            role="tab"
            aria-selected={activeMode === 'ocr'}
            onClick={() => setActiveMode('ocr')}
            disabled={isProcessing}
            className={`btn btn-sm border-none rounded-full text-xs font-semibold tracking-wide transition-smooth cursor-pointer ${
              activeMode === 'ocr'
                ? 'btn-primary shadow-xs'
                : 'btn-ghost text-base-content/75 hover:text-base-content'
            }`}
          >
            Phụ đề cứng (Video OCR)
          </button>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          {/* Minimal Hardware Pill */}
          <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-base-300/70 border border-base-content/15 text-[11px] text-base-content/85 font-mono">
            <span
              className={`w-2 h-2 rounded-full ${
                hasCuda ? 'bg-primary shadow-[0_0_6px_rgba(149,168,87,0.6)]' : 'bg-base-content/40'
              }`}
            />
            <span className="truncate max-w-[140px]">{gpuName}</span>
            {hasCuda && (
              <span className="text-[10px] uppercase font-mono px-1.5 py-0.2 rounded-full bg-primary text-primary-content font-semibold">
                CUDA
              </span>
            )}
          </div>

          {/* Primary Start / Cancel Execution Action */}
          {!isProcessing ? (
            <button
              type="button"
              onClick={handleExecute}
              disabled={!project}
              className="btn btn-sm btn-primary rounded-full font-semibold shadow-xs cursor-pointer disabled:cursor-not-allowed"
            >
              Bắt đầu trích xuất ({activeMode.toUpperCase()})
            </button>
          ) : (
            <div className="flex items-center gap-1.5">
              <button
                type="button"
                disabled
                className="btn btn-sm btn-primary rounded-full font-medium flex items-center gap-1.5 cursor-wait"
              >
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>Đang xử lý {activeMode.toUpperCase()}...</span>
              </button>

              <button
                type="button"
                onClick={onCancelJob}
                className="btn btn-sm btn-error btn-outline rounded-full font-medium cursor-pointer"
                title="Dừng tác vụ đang chạy"
              >
                Hủy tác vụ
              </button>
            </div>
          )}
        </div>
      </div>

      {/* 2. Core Parameter Controls (DaisyUI Select & Range) */}
      <div className="mt-3">
        {activeMode === 'stt' ? (
          /* ================= STT Core Controls ================= */
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
            {/* Model Selection */}
            <div>
              <label htmlFor="stt-model-select" className="block text-[11px] font-semibold text-base-content/85 mb-1">
                Mô hình Whisper
              </label>
              <select
                id="stt-model-select"
                value={sttModel}
                onChange={(e) => setSttModel(e.target.value as STTJobRequest['model_size'])}
                disabled={isProcessing}
                className="select select-bordered select-sm w-full bg-base-100 border-base-content/25 text-base-content text-xs focus:border-primary cursor-pointer"
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
              <label htmlFor="stt-lang-select" className="block text-[11px] font-semibold text-base-content/85 mb-1">
                Ngôn ngữ giọng nói
              </label>
              <select
                id="stt-lang-select"
                value={sttLanguage}
                onChange={(e) => setSttLanguage(e.target.value)}
                disabled={isProcessing}
                className="select select-bordered select-sm w-full bg-base-100 border-base-content/25 text-base-content text-xs focus:border-primary cursor-pointer"
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
              <label htmlFor="stt-device-select" className="block text-[11px] font-semibold text-base-content/85 mb-1">
                Thiết bị xử lý
              </label>
              <select
                id="stt-device-select"
                value={sttDevice}
                onChange={(e) => setSttDevice(e.target.value as 'auto' | 'cuda' | 'cpu')}
                disabled={isProcessing}
                className="select select-bordered select-sm w-full bg-base-100 border-base-content/25 text-base-content text-xs focus:border-primary cursor-pointer"
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
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
            {/* OCR Engine Selection */}
            <div>
              <label htmlFor="ocr-engine-select" className="block text-[11px] font-semibold text-base-content/85 mb-1">
                Công cụ OCR
              </label>
              <select
                id="ocr-engine-select"
                value={ocrEngine}
                onChange={(e) => setOcrEngine(e.target.value as OCRJobRequest['engine'])}
                disabled={isProcessing}
                className="select select-bordered select-sm w-full bg-base-100 border-base-content/25 text-base-content text-xs focus:border-primary cursor-pointer"
              >
                <option value="auto">Tự động (Khuyên dùng)</option>
                <option value="rapidocr">RapidOCR (ONNX - Nhanh)</option>
                <option value="vietocr">RapidOCR + VietOCR (Chuẩn dấu Việt)</option>
                <option value="gemini">Gemini Flash Vision (Cloud AI)</option>
              </select>
            </div>

            {/* OCR Language Selector */}
            <div>
              <label htmlFor="ocr-lang-select" className="block text-[11px] font-semibold text-base-content/85 mb-1">
                Ngôn ngữ chữ viết
              </label>
              <select
                id="ocr-lang-select"
                value={ocrLanguage}
                onChange={(e) => setOcrLanguage(e.target.value as OCRJobRequest['language'])}
                disabled={isProcessing}
                className="select select-bordered select-sm w-full bg-base-100 border-base-content/25 text-base-content text-xs focus:border-primary cursor-pointer"
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
              <div className="flex justify-between items-center mb-1">
                <label htmlFor="ocr-fps-range" className="text-[11px] font-semibold text-base-content/85">
                  Tần suất quét FPS
                </label>
                <span className="font-mono text-[11px] px-1.5 py-0.5 rounded bg-base-300 border border-base-content/20 text-primary font-semibold">
                  {ocrSampleFps.toFixed(1)} FPS
                </span>
              </div>
              <div className="pt-1.5">
                <input
                  id="ocr-fps-range"
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

      {/* 3. Collapsible Advanced Settings & Cloud Key Drawer */}
      <div className="mt-3 pt-2.5 border-t border-base-content/15">
        <button
          type="button"
          aria-expanded={showAdvanced}
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="btn btn-sm btn-ghost rounded-full text-[11px] font-semibold uppercase tracking-wider text-primary hover:text-base-content gap-1.5 cursor-pointer px-3"
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
          <div className="mt-2.5 p-3 rounded-xl bg-base-300/50 border border-base-content/15 space-y-3">
            {/* STT Advanced Parameters */}
            {activeMode === 'stt' && (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <label className="flex items-center gap-2 cursor-pointer select-none">
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

                <label className="flex items-center gap-2 cursor-pointer select-none">
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
                    <label htmlFor="stt-max-chars" className="text-xs text-base-content/85">
                      Ký tự tối đa mỗi dòng
                    </label>
                    <span className="font-mono text-xs font-semibold text-primary">{sttMaxChars}</span>
                  </div>
                  <input
                    id="stt-max-chars"
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
                    <label htmlFor="stt-max-lines" className="text-xs text-base-content/85">
                      Số dòng tối đa mỗi câu
                    </label>
                    <span className="font-mono text-xs font-semibold text-primary">{sttMaxLines}</span>
                  </div>
                  <input
                    id="stt-max-lines"
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
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <label htmlFor="ocr-luma-range" className="text-xs text-base-content/85">
                      Ngưỡng lọc sáng Luma
                    </label>
                    <span className="font-mono text-xs font-semibold text-primary">{ocrLumaThreshold}</span>
                  </div>
                  <input
                    id="ocr-luma-range"
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
                    <label htmlFor="ocr-stability-range" className="text-xs text-base-content/85">
                      Khung hình ổn định
                    </label>
                    <span className="font-mono text-xs font-semibold text-primary">{ocrStabilityFrames} frames</span>
                  </div>
                  <input
                    id="ocr-stability-range"
                    type="range"
                    min="0"
                    max="4"
                    value={ocrStabilityFrames}
                    onChange={(e) => setOcrStabilityFrames(parseInt(e.target.value, 10))}
                    disabled={isProcessing}
                    className="range range-primary range-xs w-full cursor-pointer disabled:opacity-40"
                  />
                </div>

                <label className="flex items-center gap-2 cursor-pointer select-none sm:col-span-2 pt-0.5">
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
            <div className="pt-2.5 border-t border-base-content/15">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1 mb-1">
                <label htmlFor="gemini-api-key-input" className="text-xs font-semibold text-base-content/90">
                  Google Gemini API Key
                </label>
                <span className="text-[11px] text-base-content/65">
                  Dùng cho OCR Vision / AI Sửa lỗi / Dịch thuật
                </span>
              </div>
              <div className="relative">
                <input
                  id="gemini-api-key-input"
                  type={showApiKey ? 'text' : 'password'}
                  placeholder="AIzaSy..."
                  value={geminiApiKey}
                  onChange={(e) => handleApiKeyChange(e.target.value)}
                  disabled={isProcessing}
                  className="input input-bordered input-sm w-full pr-9 bg-base-100 border-base-content/25 font-mono text-xs text-base-content placeholder:text-base-content/40 focus:border-primary"
                />
                <button
                  type="button"
                  onClick={() => setShowApiKey(!showApiKey)}
                  aria-label={showApiKey ? 'Ẩn khóa API' : 'Hiện khóa API'}
                  title={showApiKey ? 'Ẩn khóa' : 'Hiện khóa'}
                  className="btn btn-ghost btn-xs btn-circle absolute right-1.5 top-1/2 -translate-y-1/2 text-base-content/60 hover:text-base-content cursor-pointer"
                >
                  {showApiKey ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ControlPanel;
