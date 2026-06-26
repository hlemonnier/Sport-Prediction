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

function clearRouteChrome(): void {
  if (typeof document === "undefined") return;
  document.body.classList.remove("f1-insights-chrome");
}

export default function UiPreferencesBootstrap() {
  useEffect(() => {
    let cancelled = false;

    const sync = () => {
      applyUiPreferences(readUiPreferences());
      clearRouteChrome();
    };

    sync();

    const hydrateFromDatabase = async () => {
      if (!process.env.NEXT_PUBLIC_API_URL && !process.env.API_URL) {
        return;
      }

      try {
        const response = await getUserPreferences();
        if (cancelled) {
          return;
        }

        if (response.preferences) {
          const serverPreferences = coerceUiPreferences(response.preferences);
          writeUiPreferences(serverPreferences);
          applyUiPreferences(serverPreferences);
          clearRouteChrome();
        }

        writeUserSavings(coerceUserSavings(response.savings));
      } catch (error) {
        console.error("Failed to bootstrap user preferences", error);
      }
    };

    void hydrateFromDatabase();

    const unsubscribe = subscribeUiPreferences(sync);

    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, []);

  return null;
}
