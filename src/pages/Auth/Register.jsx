import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { registerBloger, registerCompany } from '../../api';
import '../../styles/Register.css'
export const Register = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    userType: 'regular', // 'regular', 'bloger', or 'company'
    ...(false && { niche: '' }),
    ...(false && { size: '' })
  });
  const navigate = useNavigate();

  const handleUserTypeChange = (type) => {
    setFormData({
      ...formData,
      userType: type,
      ...(type !== 'bloger' && { niche: '' }),
      ...(type !== 'company' && { size: '' })
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      if (formData.userType === 'bloger') {
        await registerBloger(formData);
      } else if (formData.userType === 'company') {
        await registerCompany(formData);
      } else {
        
      }
      navigate('/login');
    } catch (err) {
      console.error('Registration failed:', err);
    }
  };

  return (
    <div className="register-container">
      <h2>Create Account</h2>
      <div className="user-type-selector">
        <button
          className={formData.userType === 'regular' ? 'active' : ''}
          onClick={() => handleUserTypeChange('regular')}
        >
          Regular User
        </button>
        <button
          className={formData.userType === 'bloger' ? 'active' : ''}
          onClick={() => handleUserTypeChange('bloger')}
        >
          Bloger
        </button>
        <button
          className={formData.userType === 'company' ? 'active' : ''}
          onClick={() => handleUserTypeChange('company')}
        >
          Company
        </button>
      </div>

      <form onSubmit={handleSubmit}>
        <input
          name="name"
          placeholder="Full Name"
          value={formData.name}
          onChange={(e) => setFormData({...formData, [e.target.name]: e.target.value})}
          required
        />
        <input
          name="email"
          type="email"
          placeholder="Email"
          value={formData.email}
          onChange={(e) => setFormData({...formData, [e.target.name]: e.target.value})}
          required
        />
        <input
          name="password"
          type="password"
          placeholder="Password"
          value={formData.password}
          onChange={(e) => setFormData({...formData, [e.target.name]: e.target.value})}
          required
        />

        {formData.userType === 'bloger' && (
          <input
            name="niche"
            placeholder="Your Niche (e.g., Travel, Food)"
            value={formData.niche || ''}
            onChange={(e) => setFormData({...formData, niche: e.target.value})}
            required
          />
        )}

        {formData.userType === 'company' && (
          <input
            name="size"
            placeholder="Company Size"
            value={formData.size || ''}
            onChange={(e) => setFormData({...formData, size: e.target.value})}
            required
          />
        )}

        <button type="submit">Register</button>
      </form>
    </div>
  );
};