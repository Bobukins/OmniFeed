const API_BASE = process.env.REACT_APP_API_URL;

export const getCompanies = async (token) => {
  const res = await fetch(`${API_BASE}/companies`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  return res.json();
};

export const registerCompany = async (data) => {
  const res = await fetch(`${API_BASE}/register/Company`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  return res.json();
};
