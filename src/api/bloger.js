const API_BASE = process.env.REACT_APP_API_URL;

export const getBlogers = async (token) => {
  const res = await fetch(`${API_BASE}/blogers`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  return res.json();
};

export const registerBloger = async (data) => {
  const res = await fetch(`${API_BASE}/register/Bloger`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return res.json();
};
