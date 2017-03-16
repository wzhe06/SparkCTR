package com.ipinyou.offlinemodel;

import java.util.TimeZone;
import java.util.concurrent.TimeUnit;

/**
 * Created by ggstar on 12/15/16.
 */
public class TimeUtils {
    private static final long millisOffset = (long) TimeZone.getDefault().getRawOffset();

    public TimeUtils() {
    }

    public static int getSecondOfDay(long timestamp) {
        return (int)((timestamp + millisOffset) % 86400000L / 1000L);
    }

    public static int getMinuteOfDay(long timestamp) {
        return (int)((timestamp + millisOffset) % 86400000L / 60000L);
    }

    public static int getHourOfDay(long timestamp) {
        return (int)((timestamp + millisOffset) % 86400000L / 3600000L);
    }

    public static int getDayIndex(long timestamp) {
        return (int)((timestamp + millisOffset) / 86400000L);
    }

    public static int getDayOfWeek(long timestamp) {
        return (int)((timestamp + millisOffset + 259200000L) % 604800000L / 86400000L);
    }

    public static long getMondayMillis(long timestamp) {
        return timestamp - (timestamp + millisOffset + 259200000L) % 604800000L;
    }

    public static long getTimeMillis(int dayIndex, int secondOfDay) {
        return (long)dayIndex * 86400000L - millisOffset + (long)secondOfDay * 1000L;
    }

    public static long getTimeStamp(int dayIndex, int secondOfDay) {
        return (long)dayIndex * 86400000L - millisOffset + (long)secondOfDay * TimeUnit.SECONDS.toMillis(1L);
    }
}