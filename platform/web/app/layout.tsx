import type { ReactNode } from "react";
import { JetBrains_Mono, Space_Grotesk } from "next/font/google";
import Sidebar from "@/components/Sidebar";
import TopBar from "@/components/TopBar";
import ToastHost from "@/components/ToastHost";
import UiPreferencesBootstrap from "@/components/UiPreferencesBootstrap";
import "./globals.css";

const space = Space_Grotesk({ subsets: ["latin"], variable: "--font-sans" });
const mono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
});

export const metadata = {
  title: "Sport Prediction Lab",
  description: "Local quantitative research platform for sports prediction",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${space.variable} ${mono.variable}`}>
      <body>
        <UiPreferencesBootstrap />
        <ToastHost />
        <div className="bg-grid" />
        <div className="shell">
          <Sidebar />
          <div className="shell-body">
            <TopBar />
            <main className="site-main">{children}</main>
            <footer className="footer">
              Local mode — No data leaves this machine
            </footer>
          </div>
        </div>
      </body>
    </html>
  );
}
