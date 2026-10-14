export function BrandMark({ size = 36, className = "" }: { size?: number; className?: string }) {
  return (
    <span
      className={`inline-flex shrink-0 items-center justify-center rounded-full ${className}`}
      style={{ width: size, height: size, background: "#075b6e", color: "#fff" }}
      aria-hidden="true"
    >
      <svg width={size * 0.77} height={size * 0.77} viewBox="0 0 32 32" fill="none">
        <path d="M16 5.2v3.2" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
        <circle cx="16" cy="4.8" r="1.6" fill="currentColor" />
        <rect x="6.3" y="9" width="19.4" height="16.5" rx="5.1" fill="#e8f8fb" />
        <rect x="8.4" y="11.3" width="15.2" height="11.8" rx="3.3" fill="#075b6e" />
        <path d="M3.8 15v4.4M28.2 15v4.4" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" />
        <circle cx="12.6" cy="16.6" r="1.6" fill="#9de4ed" />
        <circle cx="19.4" cy="16.6" r="1.6" fill="#9de4ed" />
        <path d="M13.6 20.4h4.8" stroke="#9de4ed" strokeWidth="1.3" strokeLinecap="round" />
      </svg>
    </span>
  );
}
