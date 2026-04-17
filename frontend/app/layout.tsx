import type { Metadata } from "next";
import { Manrope, Sora, Space_Grotesk } from "next/font/google";
import "./globals.css";

const bodyFont = Manrope({
  subsets: ["latin"],
  variable: "--font-body"
});

const displayFont = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-display"
});

const heroFont = Sora({
  subsets: ["latin"],
  variable: "--font-hero"
});

export const metadata: Metadata = {
  title: "Pneumonia AI Screening",
  description: "Modern AI-assisted chest X-ray screening and review workspace"
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${bodyFont.variable} ${displayFont.variable} ${heroFont.variable}`}>{children}</body>
    </html>
  );
}
