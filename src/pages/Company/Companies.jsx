import { useState, useEffect, useContext } from 'react';
import { AuthContext } from '../../context/AuthContext';
import { getCompanies } from '../../api/company';

export const Companies = () => {
  const [companies, setCompanies] = useState([]);
  const { token } = useContext(AuthContext);

  useEffect(() => {
    if (token) {
      getCompanies(token)
        .then(setCompanies)
        .catch(console.error);
    }
  }, [token]);

  return (
    <div>
      <h1>Companies</h1>
      <ul>
        {companies.map((company) => (
          <li key={company.id}>
            <h3>{company.name}</h3>
            <p>Size: {company.size}</p>
          </li>
        ))}
      </ul>
    </div>
  );
};