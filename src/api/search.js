const API_BASE = process.env.REACT_APP_API_URL;

const handleApiResponse = async (response) => {
  const contentType = response.headers.get("content-type") || "";

  if (contentType.includes("text/html")) {
    const text = await response.text();

    const errorMatch =
      text.match(/<title>(.*?)<\/title>/i) || text.match(/<h1>(.*?)<\/h1>/i);
    const errorMessage = errorMatch
      ? errorMatch[1]
      : "Server returned HTML error page";

    throw new Error(errorMessage);
  }

  if (!contentType.includes("application/json")) {
    throw new Error(`Unexpected content type: ${contentType}`);
  }

  return response.json();
};

export const searchBlogers = async (query, token) => {
  try {
    const response = await fetch(
      `${API_BASE}/search/blogers?q=${encodeURIComponent(query)}`,
      {
        headers: {
          // authorization: `Bearer ${token}`,
          Accept: "application/json",
        },
        credentials: "include",
      }
    );

    return await handleApiResponse(response);
  } catch (error) {
    console.error("Search failed:", error);
    throw new Error(`Search failed: ${error.message}`);
  }
};

export const searchCompanies = async (query, token) => {
  try {
    const response = await fetch(
      `${API_BASE}/search/companies?q=${encodeURIComponent(query)}`,
      {
        headers: {
          // authorization: `Bearer ${token}`,
          // this line is for auth of user, if you decide to have search work only for auth users
          Accept: "application/json",
        },
        credentials: "include",
      }
    );

    return await handleApiResponse(response);
  } catch (error) {
    console.error("Search failed:", error);
    throw new Error(`Search failed: ${error.message}`);
  }
};
