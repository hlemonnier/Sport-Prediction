import type { ReactNode } from "react";
import { Inter, JetBrains_Mono } from "next/font/google";
import Sidebar from "@/components/Sidebar";
import TopBar from "@/components/TopBar";
import ToastHost from "@/components/ToastHost";
import UiPreferencesBootstrap from "@/components/UiPreferencesBootstrap";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-sans" });
const mono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
});

export const metadata = {
  title: "Sport Prediction Lab",
  description: "Local quantitative research system for sports prediction",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={`${inter.variable} ${mono.variable}`}>
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
              Runtime local — Public F1 APIs enabled
            </footer>
          </div>
        </div>
      </body>
    </html>
  );
}
