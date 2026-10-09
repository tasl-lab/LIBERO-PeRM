# Deprecated — use server_pi05.sh with a runfile yaml instead.
# Kept for muscle-memory; delegates to the new unified launcher.
#
# New form:
#   bash server_pi05.sh run_profile/ZeroShot/full_pi05.yaml
exec bash "$(dirname "$0")/server_pi05.sh" run_profile/ZeroShot/full_pi05.yaml "$@"
