import { defineStore } from "pinia";
import { batch, type BatchStartRequest, type BatchStatus } from "../api";

export const useBatchStore = defineStore("batch", {
  state: () => ({
    jobId: null as string | null,
    status: null as BatchStatus | null,
    error: null as string | null,
    polling: false as boolean,
  }),
  actions: {
    async start(req: BatchStartRequest) {
      this.error = null;
      this.status = null;
      try {
        const r = await batch.start(req);
        this.jobId = r.job_id;
        this.poll();
      } catch (e: any) {
        this.error = e.message ?? String(e);
      }
    },
    async poll() {
      if (!this.jobId) return;
      this.polling = true;
      const id = this.jobId;
      while (this.polling && this.jobId === id) {
        try {
          this.status = await batch.status(id);
          if (this.status.state === "done" || this.status.state === "failed") break;
        } catch (e: any) {
          this.error = e.message ?? String(e);
          break;
        }
        await new Promise(r => setTimeout(r, 1000));
      }
      this.polling = false;
    },
    stop() { this.polling = false; },
  },
});
