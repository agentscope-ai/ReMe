const DAILY_CRON = /^(\d{1,2})\s+(\d{1,2})\s+\*\s+\*\s+\*$/;

export function nextDailyRun(
  cron: string,
  timezone: string,
  now = new Date(),
): Date {
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
  const formatter = new Intl.DateTimeFormat("en-US", {
    timeZone: timezone,
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "numeric",
    minute: "numeric",
    hourCycle: "h23",
  });
  const next = new Date(now.getTime() - 26 * 60 * 60 * 1000);
  next.setUTCSeconds(0, 0);
  const scheduledDays = new Set<string>();
  for (let checked = 0; checked < 5 * 24 * 60; checked += 1) {
    const parts = formatter.formatToParts(next);
    const part = (type: Intl.DateTimeFormatPartTypes): string | undefined =>
      parts.find((candidate) => candidate.type === type)?.value;
    const candidateHour = Number(part("hour"));
    const candidateMinute = Number(part("minute"));
    if (candidateHour === hour && candidateMinute === minute) {
      const day = `${part("year")}-${part("month")}-${part("day")}`;
      if (!scheduledDays.has(day)) {
        scheduledDays.add(day);
        if (next.getTime() > now.getTime()) return next;
      }
    }
    next.setUTCMinutes(next.getUTCMinutes() + 1);
  }
  throw new Error("dreamCron has no occurrence in the scheduling window");
}
