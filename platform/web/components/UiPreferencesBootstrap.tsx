"use client";

import { useEffect } from "react";
import {
  applyUiPreferences,
  readUiPreferences,
  subscribeUiPreferences,
} from "@/lib/uiPreferences";

export default function UiPreferencesBootstrap() {
  useEffect(() => {
    const sync = () => {
      applyUiPreferences(readUiPreferences());
    };

    sync();
    const unsubscribe = subscribeUiPreferences(sync);

    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const handleSystemTheme = () => {
      const prefs = readUiPreferences();
      if (prefs.themeMode === "system") {
        applyUiPreferences(prefs);
      }
    };

    if (typeof media.addEventListener === "function") {
      media.addEventListener("change", handleSystemTheme);
    } else {
      media.addListener(handleSystemTheme);
    }

    return () => {
      unsubscribe();
      if (typeof media.removeEventListener === "function") {
        media.removeEventListener("change", handleSystemTheme);
      } else {
        media.removeListener(handleSystemTheme);
      }
    };
  }, []);

  return null;
}
