#!/bin/sh
set -eu

export SUPERSET_SECRET_KEY="${SUPERSET_SECRET_KEY:-diplom-superset-secret-key}"
export SUPERSET_ADMIN_USERNAME="${SUPERSET_ADMIN_USERNAME:-admin}"
export SUPERSET_ADMIN_FIRSTNAME="${SUPERSET_ADMIN_FIRSTNAME:-Diplom}"
export SUPERSET_ADMIN_LASTNAME="${SUPERSET_ADMIN_LASTNAME:-Admin}"
export SUPERSET_ADMIN_EMAIL="${SUPERSET_ADMIN_EMAIL:-admin@example.com}"
export SUPERSET_ADMIN_PASSWORD="${SUPERSET_ADMIN_PASSWORD:-adminadminadmin}"

superset db upgrade

superset fab create-admin \
  --username "$SUPERSET_ADMIN_USERNAME" \
  --firstname "$SUPERSET_ADMIN_FIRSTNAME" \
  --lastname "$SUPERSET_ADMIN_LASTNAME" \
  --email "$SUPERSET_ADMIN_EMAIL" \
  --password "$SUPERSET_ADMIN_PASSWORD" || true

superset init

exec superset run -h 0.0.0.0 -p 8088 --with-threads
