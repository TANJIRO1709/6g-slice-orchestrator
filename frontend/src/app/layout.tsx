import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "6G Slice Orchestrator — Dashboard",
  description: "Real-time monitoring for AI-powered 6G network slice management",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">{children}</body>
    </html>
  );
}
