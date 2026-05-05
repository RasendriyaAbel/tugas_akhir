import { NextResponse } from "next/server";

import { fetchLatestBlynkDataWithMeta } from "@/lib/blynk";
import { getNextMockBlynkData } from "@/lib/mockData";
import { fetchLatestThingsBoardDataWithMeta } from "@/lib/thingsboard";
import type { LatestBlynkResponse } from "@/types/nilm";

export const dynamic = "force-dynamic";

export async function GET() {
  const useDummyBlynk = process.env.USE_DUMMY_BLYNK === "true";
  const dataSource = (process.env.NILM_DATA_SOURCE || "thingsboard").trim().toLowerCase();
  const token = process.env.BLYNK_AUTH_TOKEN?.trim();

  if (useDummyBlynk) {
    const data = await getNextMockBlynkData();

    return NextResponse.json({
      success: true,
      data,
      source: "dummy",
      last_updated: data.timestamp,
      error: "Mode simulasi aktif. Data sensor saat ini menggunakan dummy live dari label model.",
    } satisfies LatestBlynkResponse);
  }

  if (dataSource === "thingsboard") {
    try {
      const result = await fetchLatestThingsBoardDataWithMeta();

      return NextResponse.json({
        success: true,
        data: result.data,
        source: "thingsboard",
        last_updated: result.data.timestamp,
        error: result.notice,
      } satisfies LatestBlynkResponse);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown ThingsBoard error";

      return NextResponse.json(
        {
          success: false,
          data: null,
          source: "thingsboard",
          last_updated: new Date().toISOString(),
          error: `ThingsBoard connection error: ${message}`,
        } satisfies LatestBlynkResponse,
        { status: 502 },
      );
    }
  }

  if (!token) {
    return NextResponse.json(
      {
        success: false,
        data: null,
        source: "blynk",
        last_updated: new Date().toISOString(),
        error: "BLYNK_AUTH_TOKEN belum diatur.",
      } satisfies LatestBlynkResponse,
      { status: 503 },
    );
  }

  try {
    const result = await fetchLatestBlynkDataWithMeta(token);

    return NextResponse.json({
      success: true,
      data: result.data,
      source: "blynk",
      last_updated: result.data.timestamp,
      error: result.notice,
    } satisfies LatestBlynkResponse);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown Blynk error";

    return NextResponse.json(
      {
        success: false,
        data: null,
        source: "blynk",
        last_updated: new Date().toISOString(),
        error: `Blynk connection error: ${message}`,
      } satisfies LatestBlynkResponse,
      { status: 502 },
    );
  }
}
