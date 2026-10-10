import "./globals.css";
import Nav from "../components/Nav";

export const metadata = {
  title: "Vault Document System",
  description: "Chat with your documents, with verified citations.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <Nav />
        <main className="page">{children}</main>
      </body>
    </html>
  );
}
