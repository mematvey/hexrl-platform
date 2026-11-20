# Образ Tempo без libc: подходит только статически собранный busybox
FROM busybox:1.37.0-uclibc AS healthcheck

FROM grafana/tempo:2.7.2
COPY --from=healthcheck /bin/busybox /busybox
