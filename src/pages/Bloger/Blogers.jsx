import { useState, useEffect, useContext } from 'react';
import { AuthContext } from '../../context/AuthContext';
import { getBlogers } from '../../api/bloger';

export const Blogers = () => {
  const [blogers, setBlogers] = useState([]);
  const { token } = useContext(AuthContext);

  useEffect(() => {
    if (token) {
      getBlogers(token)
        .then(setBlogers)
        .catch(console.error);
    }
  }, [token]);

  return (
    <div>
      <h1>Blogers</h1>
      <ul>
        {blogers.map((bloger) => (
          <li key={bloger.id}>
            <h3>{bloger.name}</h3>
            <p>{bloger.email}</p>
          </li>
        ))}
      </ul>
    </div>
  );
};