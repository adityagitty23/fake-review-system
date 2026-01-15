import { User, AnalysisHistoryItem } from "../types";

const USERS_KEY = 'frs_users';
const CURRENT_USER_KEY = 'frs_current_user';
const HISTORY_PREFIX = 'frs_history_';

// --- AUTHENTICATION ---

export const signUp = (email: string, password: string, name: string): User => {
  const usersStr = localStorage.getItem(USERS_KEY);
  const users = usersStr ? JSON.parse(usersStr) : {};

  if (users[email]) {
    throw new Error("User already exists");
  }

  // In a real app, hash the password!
  users[email] = { email, password, name };
  localStorage.setItem(USERS_KEY, JSON.stringify(users));

  const user = { email, name };
  localStorage.setItem(CURRENT_USER_KEY, JSON.stringify(user));
  return user;
};

export const signIn = (email: string, password: string): User => {
  const usersStr = localStorage.getItem(USERS_KEY);
  const users = usersStr ? JSON.parse(usersStr) : {};
  const account = users[email];

  if (!account || account.password !== password) {
    throw new Error("Invalid email or password");
  }

  const user = { email, name: account.name };
  localStorage.setItem(CURRENT_USER_KEY, JSON.stringify(user));
  return user;
};

export const signOut = () => {
  localStorage.removeItem(CURRENT_USER_KEY);
};

export const getCurrentUser = (): User | null => {
  const userStr = localStorage.getItem(CURRENT_USER_KEY);
  return userStr ? JSON.parse(userStr) : null;
};

// --- HISTORY ---

export const saveHistory = (email: string, item: AnalysisHistoryItem) => {
  const key = `${HISTORY_PREFIX}${email}`;
  const historyStr = localStorage.getItem(key);
  const history = historyStr ? JSON.parse(historyStr) : [];
  
  history.unshift(item); // Add to top
  localStorage.setItem(key, JSON.stringify(history));
};

export const getHistory = (email: string): AnalysisHistoryItem[] => {
  const key = `${HISTORY_PREFIX}${email}`;
  const historyStr = localStorage.getItem(key);
  return historyStr ? JSON.parse(historyStr) : [];
};
