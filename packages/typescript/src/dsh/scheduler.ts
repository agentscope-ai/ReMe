const DAILY_CRON = /^(\d{1,2})\s+(\d{1,2})\s+\*\s+\*\s+\*$/;

export function nextDailyRun(cron: string, now = new Date()): Date {
  const match = DAILY_CRON.exec(String(cron || "").trim());
  if (!match) {
    throw new Error(
      "dreamCron must use the daily form '<minute> <hour> * * *'",
    );
  }
  const minute = Number(match[1]);
  const hour = Number(match[2]);
  if (minute > 59 || hour > 23)
    throw new Error("dreamCron contains an invalid hour or minute");
  const next = new Date(now.getTime());
  next.setHours(hour, minute, 0, 0);
  if (next.getTime() <= now.getTime()) next.setDate(next.getDate() + 1);
  return next;
}
