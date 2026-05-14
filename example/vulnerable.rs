fn from(mut timestamp: Timestamp) -> std::time::SystemTime {
    timestamp.normalize();
    let system_time = if timestamp.seconds >= 0 {
        std::time::UNIX_EPOCH + time::Duration::from_secs(timestamp.seconds as u64)
    } else {
        std::time::UNIX_EPOCH - time::Duration::from_secs((-timestamp.seconds) as u64)
    };

    system_time + time::Duration::from_nanos(timestamp.nanos as u64)
}