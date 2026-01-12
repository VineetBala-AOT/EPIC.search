import {
  Box,
  Typography,
  Button,
  Menu,
  IconButton,
  MenuItem,
} from "@mui/material";
import { useAuth } from "react-oidc-context";
import AccountCircleIcon from "@mui/icons-material/AccountCircle";
import { theme } from "@/styles/theme";
import KeyboardArrowDownIcon from "@mui/icons-material/KeyboardArrowDown";
import { useState } from "react";
import { OidcConfig } from "@/utils/config";
import { BCDesignTokens } from "epic.theme";
import { useNavigate } from "@tanstack/react-router";

export default function AppBarActions() {
  const auth = useAuth();
  const navigate = useNavigate();
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const open = Boolean(anchorEl);
  const handleClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  };
  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleNavigate = (path: string) => {
    setAnchorEl(null);
    navigate({ to: path });
  };

  return (
    <>
      {auth.isAuthenticated ? (
        <>
          <Box id="menu-appbar" display={"flex"} onClick={handleClick}>
            <Typography
              variant="body2"
              color="primary"
              onClick={handleClick}
              sx={{ cursor: "pointer" }}
            >
              Hi, {auth.user?.profile.given_name}
            </Typography>
            <IconButton size="small" sx={{ m: 0, p: 0 }}>
              <KeyboardArrowDownIcon
                fontSize="small"
                htmlColor={theme.palette.grey[900]}
              />
            </IconButton>
          </Box>
          <AccountCircleIcon
            fontSize="large"
            htmlColor={theme.palette.grey[900]}
            sx={{ marginLeft: "0.25rem" }}
          />
          <Menu
            id="menu-appbar"
            aria-labelledby="menu-appbar"
            open={open}
            anchorEl={anchorEl}
            onClose={handleClose}
            anchorOrigin={{
              vertical: "top",
              horizontal: "right",
            }}
            transformOrigin={{
              vertical: "top",
              horizontal: "right",
            }}
          >
            <MenuItem
              onClick={() => {
                handleNavigate("/logout");
              }}
            >
              Sign Out
            </MenuItem>
          </Menu>
        </>
      ) : (
        <Button
          variant="text"
          onClick={() =>
            auth.signinRedirect({
              redirect_uri: `${OidcConfig.redirect_uri}${window.location.search}`,
              extraQueryParams: {
                kc_idp_hint: OidcConfig.kc_idp_hint,
              },
            })
          }
          sx={{
            color: BCDesignTokens.themeGray100,
            border: `2px solid ${theme.palette.grey[700]}`,
          }}
        >
          Login
        </Button>
      )}
    </>
  );
}
