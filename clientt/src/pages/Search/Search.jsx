import { useState, useEffect } from "react";
import { searchBlogers, searchCompanies } from "../../api/search.js";
import TopNav from "../../components/TopNav.jsx";
import { useLocation } from "react-router-dom";
import Navbar from "../../components/Navbar";

export const Search = () => {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [searchType, setSearchType] = useState("blogers");

  const handleSearch = async () => {
    setError(null);
    setIsLoading(true);

    try {
      // const token = localStorage.getItem("token"); // or auth token source
      const searchFn =
        searchType === "blogers" ? searchBlogers : searchCompanies;
      const data = await searchFn(query);
      const maybeResults = data?.results || data;
      setResults(Array.isArray(maybeResults) ? maybeResults : []);
    } catch (err) {
      setError(err.message);
      setResults([]);
    } finally {
      setIsLoading(false);
    }
  };
  const location = useLocation();

  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const q = params.get("q");
    if (q) {
      setQuery(q);
      handleSearch();
    }
  }, [location.search]);
  return (
    <div>

    
      <Navbar />
    <div className="search-container">
      <div style={{ display: "flex", flexDirection: "row" }}>
        <select
          value={searchType}
          onChange={(e) => setSearchType(e.target.value)}>
          <option value="blogers">Blogers</option>
          <option value="companies">Companies</option>
        </select>

        <TopNav query={query} setQuery={setQuery} onSearch={handleSearch} />
      </div>

      {error && <div className="error-message">Error: {error}</div>}

      <div className="results">
        {results.map((item) => (
          <div key={item.id} className="result-item">
            <h3>{item.name}</h3>
            <p>{item.description || item.email || ""}</p>
          </div>
        ))}
      </div>
    </div>
    </div>
  );
};
