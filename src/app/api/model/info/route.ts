import { NextResponse } from "next/server";

import { readTrainedModelInfo } from "@/lib/modelInfo";
import type { TrainedModelInfoResponse } from "@/types/nilm";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const data = await readTrainedModelInfo();

    return NextResponse.json({
      success: true,
      data,
    } satisfies TrainedModelInfoResponse);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown model inspection error";

    return NextResponse.json(
      {
        success: false,
        data: null,
        error: `Gagal membaca model terlatih: ${message}`,
      } satisfies TrainedModelInfoResponse,
      { status: 500 },
    );
  }
}
