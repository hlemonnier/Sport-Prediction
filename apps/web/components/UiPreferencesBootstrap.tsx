"use client";

import { useEffect } from "react";
import { getUserPreferences } from "@/lib/api";
import {
  applyUiPreferences,
  coerceUiPreferences,
  coerceUserSavings,
  readUiPreferences,
  subscribeUiPreferences,
  writeUserSavings,
  writeUiPreferences,
} from "@/lib/uiPreferences";

export default function UiPreferencesBootstrap() {
  useEffect(() => {
    let cancelled = false;

    const sync = () => {
      applyUiPreferences(readUiPreferences());
    };

    sync();

    const hydrateFromDatabase = async () => {
      try {
        const response = await getUserPreferences();
        if (cancelled) {
          return;
        }

        if (response.preferences) {
          const serverPreferences = coerceUiPreferences(response.preferences);
          writeUiPreferences(serverPreferences);
          applyUiPreferences(serverPreferences);
        }

        writeUserSavings(coerceUserSavings(response.savings));
      } catch (error) {
        console.error("Failed to bootstrap user preferences", error);
      }
    };

    void hydrateFromDatabase();

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
      cancelled = true;
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
