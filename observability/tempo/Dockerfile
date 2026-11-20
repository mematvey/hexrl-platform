FROM busybox:1.37.0 AS healthcheck

FROM grafana/tempo:2.7.2
COPY --from=healthcheck /bin/busybox /busybox
