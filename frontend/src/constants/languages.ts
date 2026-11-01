export interface LanguageOption {
  code: string;
  label: string;
  name?: string;
  flag: string;
}

export const TRANSLATE_TARGET_LANGUAGES: LanguageOption[] = [
  { code: 'vi', label: 'Tiếng Việt', flag: '🇻🇳' },
  { code: 'en', label: 'English', flag: '🇺🇸' },
  { code: 'zh', label: '中文 (Chinese)', flag: '🇨🇳' },
  { code: 'ja', label: '日本語 (Japanese)', flag: '🇯🇵' },
  { code: 'ko', label: '한국어 (Korean)', flag: '🇰🇷' },
  { code: 'fr', label: 'Français (French)', flag: '🇫🇷' },
  { code: 'de', label: 'Deutsch (German)', flag: '🇩🇪' },
  { code: 'es', label: 'Español (Spanish)', flag: '🇪🇸' },
  { code: 'ru', label: 'Русский (Russian)', flag: '🇷🇺' },
  { code: 'th', label: 'ไทย (Thai)', flag: '🇹🇭' },
];

export const AUDIO_LANGUAGES: LanguageOption[] = [
  { code: 'auto', label: 'Auto (Tự động nhận diện)', flag: '' },
  { code: 'vi', label: 'Tiếng Việt', flag: '🇻🇳' },
  { code: 'en', label: 'Tiếng Anh (English)', flag: '🇺🇸' },
  { code: 'zh', label: 'Tiếng Trung (Chinese)', flag: '🇨🇳' },
  { code: 'ja', label: 'Tiếng Nhật (Japanese)', flag: '🇯🇵' },
  { code: 'ko', label: 'Tiếng Hàn (Korean)', flag: '🇰🇷' },
  { code: 'fr', label: 'Tiếng Pháp (French)', flag: '🇫🇷' },
  { code: 'de', label: 'Tiếng Đức (German)', flag: '🇩🇪' },
  { code: 'es', label: 'Tiếng Tây Ban Nha (Spanish)', flag: '🇪🇸' },
  { code: 'ru', label: 'Tiếng Nga (Russian)', flag: '🇷🇺' },
];

export const OCR_LANGUAGES: LanguageOption[] = [
  { code: 'auto', label: 'Auto (Tự động nhận diện)', flag: '' },
  { code: 'vi', label: 'Tiếng Việt', flag: '🇻🇳' },
  { code: 'en', label: 'Tiếng Anh (English)', flag: '🇺🇸' },
  { code: 'ch', label: 'Tiếng Trung (Chinese)', flag: '🇨🇳' },
  { code: 'ja', label: 'Tiếng Nhật (Japanese)', flag: '🇯🇵' },
  { code: 'ko', label: 'Tiếng Hàn (Korean)', flag: '🇰🇷' },
  { code: 'multi', label: 'Đa ngôn ngữ (Multilingual)', flag: '' },
];
