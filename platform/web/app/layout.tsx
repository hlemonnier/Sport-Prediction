import type { ReactNode } from "react";
import { Space_Grotesk, Unbounded } from "next/font/google";
import Sidebar from "@/components/Sidebar";
import TopBar from "@/components/TopBar";
import "./globals.css";

const space = Space_Grotesk({ subsets: ["latin"], variable: "--font-sans" });
const unbounded = Unbounded({
  subsets: ["latin"],
  weight: ["400", "600", "700"],
  variable: "--font-display",
});

export const metadata = {
  title: "Sport Prediction Lab",
  description: "Local quant research platform for sport analytics",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="fr" className={`${space.variable} ${unbounded.variable}`}>
      <body>
        <div className="bg-grid" />
        <div className="shell">
          <Sidebar />
          <div className="shell-body">
            <TopBar />
            <main className="site-main">{children}</main>
            <footer className="footer">
              Local-first research. Data never leaves this machine.
            </footer>
          </div>
        </div>
      </body>
    </html>
  );
}
