const statusElement = document.getElementById("connection-status");
const connectButton = document.getElementById("connect-spotify-button");
const logoutButton = document.getElementById("logout-button");

function setConnected(displayName) {
  statusElement.textContent = `Connected as ${displayName}`;
  connectButton.hidden = true;
  logoutButton.hidden = false;
}

function setDisconnected() {
  statusElement.textContent = "Not connected.";
  connectButton.hidden = false;
  logoutButton.hidden = true;
}

function clearUrlQueryString() {
  const cleanUrl = `${window.location.pathname}${window.location.hash}`;
  window.history.replaceState({}, document.title, cleanUrl);
}

async function parseJsonSafe(response) {
  const body = await response.text();
  if (!body) {
    return null;
  }

  try {
    return JSON.parse(body);
  } catch (error) {
    return null;
  }
}

function getErrorMessage(payload, fallback) {
  if (!payload || typeof payload !== "object") {
    return fallback;
  }

  if (typeof payload.detail === "string" && payload.detail) {
    return payload.detail;
  }

  if (typeof payload.message === "string" && payload.message) {
    return payload.message;
  }

  return fallback;
}

async function apiFetch(path, options = {}) {
  let response;
  try {
    response = await fetch(path, {
      credentials: "same-origin",
      ...options,
    });
  } catch (error) {
    throw { status: 0, message: "Network request failed" };
  }

  const payload = await parseJsonSafe(response);
  if (!response.ok) {
    const fallback = `Request failed with status ${response.status}`;
    throw {
      status: response.status,
      message: getErrorMessage(payload, fallback),
    };
  }

  return payload;
}

async function apiMe() {
  return apiFetch("/api/me", { method: "GET" });
}

async function refreshConnectionStatus() {
  try {
    const profile = await apiMe();
    const displayName =
      profile && typeof profile.display_name === "string" && profile.display_name
        ? profile.display_name
        : "Spotify user";
    setConnected(displayName);
  } catch (error) {
    setDisconnected();
  }
}

async function connectSpotify() {
  window.location.assign("/api/auth/spotify/login");
}

async function logoutSpotify() {
  try {
    await apiFetch("/api/auth/logout", { method: "GET" });
  } finally {
    await refreshConnectionStatus();
  }
}

connectButton.addEventListener("click", () => {
  void connectSpotify();
});
logoutButton.addEventListener("click", logoutSpotify);

async function initializeApp() {
  const params = new URLSearchParams(window.location.search);
  if (params.has("code") || params.has("state") || params.has("error")) {
    clearUrlQueryString();
  }

  await refreshConnectionStatus();
}

window.apiMe = apiMe;

void initializeApp();
