import EAOAppBar from "@/components/Shared/Header/EAOAppBar";
import PageNotFound from "@/components/Shared/PageNotFound";
import { Box } from "@mui/system";
import { createRootRouteWithContext, Outlet } from "@tanstack/react-router";
import { TanStackRouterDevtools } from "@tanstack/router-devtools";
import { AuthContextProps } from "react-oidc-context";
import { ReactQueryDevtools } from "@tanstack/react-query-devtools";

type RouterContext = {
  authentication: AuthContextProps;
};

export const Route = createRootRouteWithContext<RouterContext>()({
  component: Layout,
  notFoundComponent: PageNotFound,
});

function Layout() {
  return (
    <>
      <EAOAppBar />
      <Box minHeight={"calc(100vh - 88px)"}>
        <Outlet />
      </Box>
      <TanStackRouterDevtools position="bottom-left" />
      <ReactQueryDevtools initialIsOpen={false} />
    </>
  );
}
