const API_BASE = process.env.REACT_APP_API_URL;

export const loginBloger = async (data) => {
  const res = await fetch(`${API_BASE}/login/Bloger`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return res.json();
};

export const loginCompany = async (data) => {
  const res = await fetch(`${API_BASE}/login/company`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return res.json();
};

export const logout = async (token) => {
  await fetch(`${API_BASE}/logout`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
  });
};

export const getCurrentUser = async (token) => {
  const res = await fetch(`${API_BASE}/users/me`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  return res.json();
};
