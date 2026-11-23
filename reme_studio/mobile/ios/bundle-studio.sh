#!/bin/sh
set -eu

mobile_dir="${SRCROOT}/.."
studio_dir="${mobile_dir}/.."
studio_assets="${TARGET_BUILD_DIR}/${UNLOCALIZED_RESOURCES_FOLDER_PATH}/Studio"
studio_info="${DERIVED_FILE_DIR}/Studio-Info.plist"

# Xcode launched from Finder does not inherit the local Node version manager's PATH.
export PATH="$PATH:/opt/homebrew/bin:/usr/local/bin"
if ! command -v "${NODE_BINARY:-node}" >/dev/null 2>&1; then
    printf '%s\n' 'error: Install Node.js 22.13+ or set NODE_BINARY to its absolute path in the Xcode build settings.' >&2
    exit 1
fi
if [ ! -f "${studio_dir}/node_modules/vite/bin/vite.js" ]; then
    printf '%s\n' 'error: Install Studio dependencies with npm ci --prefix reme_studio.' >&2
    exit 1
fi

cd "${mobile_dir}"
"${NODE_BINARY:-node}" "${studio_dir}/node_modules/vite/bin/vite.js" build --mode ios --outDir "${studio_assets}" --emptyOutDir

reme_version=$(/usr/bin/sed -n 's/^__version__ = "\([^"]*\)"$/\1/p' "${studio_dir}/../reme/_version.py")
test -n "${reme_version}"
mkdir -p "${DERIVED_FILE_DIR}"
cp "${SRCROOT}/ReMeStudio/Info.plist" "${studio_info}"
/usr/bin/plutil -replace CFBundleShortVersionString -string "${reme_version}" "${studio_info}"
