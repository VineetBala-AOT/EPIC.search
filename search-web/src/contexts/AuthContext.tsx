import { createContext, useContext, useEffect, useState, ReactNode } from "react";
import { useAuth } from "react-oidc-context";
import { getUserRolesFromToken } from "@/utils";

export interface RoleContext {
  roles: string[] | null; // null = loading, [] = loaded, no roles
  isAdmin: boolean;
  isViewer: boolean;
  hasAnyRole: boolean;
  isLoading: boolean;
}

const AuthRolesContext = createContext<RoleContext>({
  roles: null,
  isAdmin: false,
  isViewer: false,
  hasAnyRole: false,
  isLoading: true,
});

export function AuthRolesProvider({ children }: { children: ReactNode }) {
  const auth = useAuth();
  const [roles, setRoles] = useState<string[] | null>(null);

  useEffect(() => {
    if (auth.isAuthenticated) {
      const token = auth.user?.access_token;
      const parsedRoles = token ? getUserRolesFromToken(token) : [];
      setRoles(parsedRoles);
    } else {
      setRoles(null);
    }
  }, [auth.isAuthenticated, auth.user]);

  const isLoading = roles === null;
  const isAdmin = roles?.includes("admin") ?? false;
  const isViewer = roles?.includes("viewer") ?? false;
  const hasAnyRole = (roles?.length ?? 0) > 0;

  return (
    <AuthRolesContext.Provider value={{ roles, isAdmin, isViewer, hasAnyRole, isLoading }}>
      {children}
    </AuthRolesContext.Provider>
  );
}

export function useRoles() {
  return useContext(AuthRolesContext);
}
